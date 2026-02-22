import copy
import time
import traceback
from datetime import datetime
import pytz

# === НАСТРОЙКИ ПРОМПТОВ (Можно вынести в отдельный файл) ===

REFLECT_PROMPT = (
    "Сделай краткую рефлексию (1–2 строки на пункт):\n"
    "✅ Сделано: что выполнено и каков результат\n"
    "📋 Соответствие плану: выполнен ли нужный шаг\n"
    "⚠️ Проблемы: ошибки, риски (или 'нет')\n"
    "🔄 Корректировка: нужно ли менять следующие шаги (да/нет + что)\n"
    "➡️ Следующий шаг: конкретное действие\n"
    "Без вступлений, строго по структуре."
)

PLAN_PROMPT = (
    "Ты — интеллектуальный агент. Составь план выполнения задачи.\n"
    "Правила:\n"
    "1. Строго нумерованный список.\n"
    "2. Каждый пункт — ровно 1 строка, суть действия.\n"
    "3. Начинай с глагола (Найти, Проверить, Отправить).\n"
    "В конце добавь строку: REQUIRES_TOOLS: YES/NO."
)

EXECUTE_PROMPT = (
    "Выполняй план шаг за шагом.\n"
    "После каждого шага будет рефлексия — используй её для корректировки.\n"
    "Когда задача решена — напиши АГЕНТ_ГОТОВ."
)

class AgentState:
    """Хранилище состояния агента между запросами."""
    def __init__(self):
        self.internal_messages = []   # История сообщений
        self.state = 'PLANNING'       # Текущее состояние (PLANNING, EXECUTING, etc.)
        self.step_counter = 0         # Счетчик шагов (для защиты от петель)
        self.tool_call_count = 0      # Количество вызовов инструментов
        self.error_count = 0          # Счетчик ошибок
        self.seen_signatures = []     # Для детекта циклов

class UniversalAgent:
    """
    Универсальный класс Агента.
    Работает по принципу конечного автомата (State Machine).
    """
    
    STEP_HARD_LIMIT = 30
    ERROR_LIMIT = 5

    def __init__(self, llm_client, tools_handler, system_prompt="Ты полезный ассистент."):
        """
        :param llm_client: Функция вызова LLM. 
                           Сигнатура: (messages, tools=None) -> dict ответа OpenAI format.
        :param tools_handler: Функция обработки инструментов. 
                              Сигнатура: (function_name, arguments) -> str result.
        :param system_prompt: Системный промпт.
        """
        self.llm_client = llm_client
        self.tools_handler = tools_handler
        self.system_prompt = system_prompt
        self.state = AgentState()
        
        # Добавляем дату в системный промпт
        tz = pytz.timezone('Europe/Moscow')
        now = datetime.now(tz)
        date_str = now.strftime(f"%Y, %B {now.day}, %A, %H:%M:%S")
        self.full_system_prompt = f"{system_prompt}\n\nТекущая дата: {date_str}"

    def log(self, msg):
        """Логирование (можно заменить на logger)"""
        print(f"[AGENT] {msg}")

    def notify(self, msg):
        """Отправка уведомлений пользователю (заглушка)"""
        print(f"-> NOTIFY: {msg}")

    def _get_model_response(self, messages, with_tools=False):
        """Обертка над вызовом модели"""
        try:
            # Добавляем системный промпт
            full_msgs = [{"role": "system", "content": self.full_system_prompt}] + messages
            return self.llm_client(full_msgs, with_tools=with_tools)
        except Exception as e:
            self.log(f"Model Error: {e}")
            return None

    def _call_tool(self, func_name, args):
        """Безопасный вызов инструмента"""
        self.log(f"Tool Call: {func_name} | Args: {str(args)[:100]}")
        try:
            result = self.tools_handler(func_name, args)
            return str(result), False
        except Exception as e:
            return f"Error in {func_name}: {e}", True

    def _check_loop(self, fn, args):
        """Проверка на зацикливание по сигнатуре вызова"""
        sig = f"{fn}:{str(args)[:50]}"
        if sig in self.state.seen_signatures:
            self.log(f"LOOP DETECTED: {sig}")
            return True
        self.state.seen_signatures.append(sig)
        return False

    # === МАШИНА СОСТОЯНИЙ ===

    def _phase_planning(self, user_query):
        """Фаза 1: Планирование"""
        self.log("Phase: PLANNING")
        self.notify("📋 Составляю план...")
        
        msgs = [{"role": "user", "content": user_query}]
        # Просим модель составить план
        resp = self._get_model_response(msgs + [{"role": "user", "content": PLAN_PROMPT}])
        
        if not resp: return "Ошибка планирования."
        
        plan_text = resp.get('content', '')
        self.log(f"Plan generated:\n{plan_text}")

        # Проверяем, нужны ли инструменты
        needs_tools = "requires_tools: yes" in plan_text.lower()
        
        if not needs_tools:
            # Если инструменты не нужны — сразу отвечаем
            self.log("Tools not required, direct answer.")
            return plan_text.split("REQUIRES_TOOLS")[0].strip()

        # Если нужны — переходим к выполнению
        self.state.internal_messages = msgs
        self.state.internal_messages.append({"role": "assistant", "content": plan_text})
        self.state.internal_messages.append({"role": "user", "content": EXECUTE_PROMPT})
        self.state.state = 'EXECUTING'
        return None  # Продолжаем работу

    def _phase_executing(self):
        """Фаза 2: Выполнение и Рефлексия"""
        self.log("Phase: EXECUTING")
        
        # Ограничитель итераций
        for iteration in range(10):
            self.state.step_counter += 1
            if self.state.step_counter > self.STEP_HARD_LIMIT:
                self.state.state = 'SUMMARIZING'
                return None

            # Запрос к модели (с включением инструментов)
            resp = self._get_model_response(self.state.internal_messages, with_tools=True)
            if not resp:
                self.state.error_count += 1
                if self.state.error_count > 3: return "Слишком много ошибок API."
                continue

            # 1. Если модель хочет вызвать функцию
            if 'tool_calls' in resp and resp['tool_calls']:
                tool_call = resp['tool_calls'][0] # Берем первый вызов
                fn = tool_call['function']['name']
                args = tool_call['function']['arguments']
                
                # Проверка на цикл
                if self._check_loop(fn, args):
                    self.state.state = 'SUMMARIZING'
                    return None

                # Вызов инструмента
                result, is_error = self._call_tool(fn, args)
                self.state.tool_call_count += 1
                
                # Добавляем результат в историю
                self.state.internal_messages.append({"role": "assistant", "tool_calls": [tool_call]})
                self.state.internal_messages.append({
                    "role": "tool", 
                    "name": fn, 
                    "content": result
                })

                # Рефлексия после вызова
                self._do_reflection()
                continue

            # 2. Если модель вернула текст
            content = resp.get('content', '')
            if content:
                self.state.internal_messages.append({"role": "assistant", "content": content})
                
                # Проверка на маркер завершения
                if "АГЕНТ_ГОТОВ" in content.upper():
                    self.log("Agent marked as READY.")
                    self.state.state = 'SUMMARIZING'
                    return None
            
            # Если пусто или нет маркера — просим продолжать
            self.state.internal_messages.append({"role": "user", "content": "Продолжай. Если готов — напиши АГЕНТ_ГОТОВ."})
        
        return None

    def _do_reflection(self):
        """Внутренняя фаза: Рефлексия"""
        # Добавляем промпт рефлексии и получаем ответ
        reflect_msg = {"role": "user", "content": REFLECT_PROMPT}
        # Временно добавляем
        temp_msgs = self.state.internal_messages + [reflect_msg]
        resp = self._get_model_response(temp_msgs)
        
        if resp and resp.get('content'):
            self.log(f"Reflection: {resp['content'][:100]}...")
            self.state.internal_messages.append(reflect_msg)
            self.state.internal_messages.append({"role": "assistant", "content": resp['content']})

    def _phase_summarizing(self):
        """Фаза 3: Итоги"""
        self.log("Phase: SUMMARIZING")
        self.notify("🏁 Формирую итоговый ответ...")
        
        # Просим модель дать финальный ответ на основе истории
        final_prompt = {
            "role": "user", 
            "content": "Задача завершена. Сформулируй краткий и вежливый ответ пользователю с результатом. Не упоминай функции."
        }
        resp = self._get_model_response(self.state.internal_messages + [final_prompt])
        
        return resp.get('content', 'Задача выполнена.') if resp else "Не удалось сформировать итог."

    def run(self, user_query):
        """Точка входа"""
        # Если это новый запрос — инициализируем планирование
        if not self.state.internal_messages:
            return self._phase_planning(user_query)
        
        # Основной цикл состояний
        while True:
            if self.state.state == 'EXECUTING':
                result = self._phase_executing()
                if result: return result
                if self.state.state == 'SUMMARIZING':
                    return self._phase_summarizing()
            
            # Механика прерываний или ожидания (упрощенно для примера)
            # В реальном приложении здесь может быть yield или callback
            time.sleep(0.1) 
            
            # Защита от бесконечного цикла while
            if self.state.step_counter > self.STEP_HARD_LIMIT:
                return self._phase_summarizing()

# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===

if __name__ == "__main__":
    # 1. Эмуляция клиента LLM (OpenAI пример)
    def mock_llm_client(messages, with_tools=False):
        # Здесь должен быть реальный вызов API (openai.chat.completions.create...)
        # Для примера вернем заглушку
        last_msg = messages[-1]['content']
        
        if "План" in last_msg:
            return {"content": "1. Проверить погоду.\nREQUIRES_TOOLS: YES"}
        elif "погода" in last_msg.lower() or "Выполняй план" in last_msg:
            # Эмуляция вызова функции
            return {
                "content": None,
                "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
            }
        elif "tool" in messages[-1].get('role', ''):
             # После получения результата погоды
             return {"content": "Погода солнечная. АГЕНТ_ГОТОВ"}
        
        return {"content": "Не понял запрос."}

    # 2. Эмуляция обработчика инструментов
    def mock_tools_handler(name, args):
        if name == "get_weather": return "25 градусов, солнечно."
        return "Unknown tool"

    # 3. Запуск
    agent = UniversalAgent(mock_llm_client, mock_tools_handler)
    print("--- Запуск Агента ---")
    result = agent.run("Узнай погоду и скажи, брать ли зонтик.")
    print(f"\n--- Результат ---\n{result}")
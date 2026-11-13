from core import Bot

AVAILABLE_MODELS: dict[str, dict[str, str]] = {
    "1": {
        "name": "Google Gemma 3 4B Instruct (США, 4B)",
        "repo_id": "google/gemma-3-4b-it",
    },
    "2": {
        "name": "Google Gemma 4 E2B Instruct (США, ~2B)",
        "repo_id": "google/gemma-4-E2B-it",
    },
    "3": {
        "name": "Microsoft Phi-4 Mini Instruct (США, 3.8B)",
        "repo_id": "microsoft/Phi-4-mini-instruct",
    },
    "4": {
        "name": "Meta Llama 3.2 3B Instruct (США, 3B)",
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
    },
    "5": {
        "name": "Mistral Ministral 3 3B Instruct (Франція, 3B)",
        "repo_id": "mistralai/Ministral-3-3B-Instruct-2512",
    },
    "6": {
        "name": "PleIAs Pleias 3B Preview [Base] (Франція, 3B)",
        "repo_id": "PleIAs/Pleias-3b-Preview",
    },
}


def choose_model() -> str | None:
    print("ОБЕРІТЬ МОДЕЛЬ ДЛЯ ЗАПУСКУ:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"[{key}] {info['name']} -> {info['repo_id']}")
    print("[0]Вихід з програми")

    while True:
        try:
            choice = input("Ваш вибір (1-6 або 0): ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        if choice == "0":
            return None
        if choice in AVAILABLE_MODELS:
            selected = AVAILABLE_MODELS[choice]
            print(f"\nОбрано: {selected['name']}")
            return selected["repo_id"]

        print("Невірний вибір. Введіть число від 1 до 6 (або 0 для виходу).")


def show() -> None:
    model_id = choose_model()
    if not model_id:
        print("Роботу програми завершено.")
        return

    print("Завантаження моделі, зачекайте...")
    print("Для завершення діалогу введіть 'вихід' або 'exit'.\n")

    try:
        bot = Bot(model_name=model_id)
        print(f"Модель '{model_id}' успішно завантажена! Використовується пристрій: {bot.device}\n")
    except Exception as e:
        print(f"Критична помилка ініціалізації [{type(e).__name__}]: {e}")
        return

    while True:
        try:
            user_input = input("Ви: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nРоботу програми перервано користувачем. До побачення!")
            break

        if not user_input:
            continue

        match user_input.lower():
            case "вихід" | "exit":
                print("Бот: До побачення! Успіхів у програмуванні!")
                break
            case _:
                try:
                    print("Бот думає...", end="\r")
                    response = bot.generate_response(user_input)
                    print(f"Бот: {response}\n")
                except Exception as e:
                    print(f"\nПомилка генерації [{type(e).__name__}]: {e}\n")


if __name__ == "__main__":
    show()
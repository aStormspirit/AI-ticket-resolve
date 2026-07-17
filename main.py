from bot.pipeline import process_ticket


if __name__ == "__main__":
    sample = "Не могу войти в аккаунт, сбросьте пароль на user@example.com"
    print(process_ticket(sample, ticket_id="T-001"))

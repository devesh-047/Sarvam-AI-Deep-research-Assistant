import asyncio
from app.multilingual.response_localizer import localize_response

async def main():
    text = "This is a test to check if it translates to Bengali.\n\n**Sources:**\n- [S1] Test"
    res = await localize_response(text, "bn")
    print("Localized response:")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())

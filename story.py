import os
import requests
import openai
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
IDEOGRAM_API_KEY = os.environ.get("IDEOGRAM_API_KEY")

# ストーリー生成
def generate_page_story(main_character, main_character_name, theme, sub_characters, storyline, target_age, page_number, total_pages, previous_content=""):
    openai.api_key = OPENAI_API_KEY

    if page_number == total_pages:
        ending_instruction = "このページでストーリーを完結させてください。"
    elif page_number == total_pages - 1:
        ending_instruction = "次のページが最後のページになるように内容を調整してください。"
    else:
        ending_instruction = "次のページに続く内容にしてください。「…。」のように文章を終わらせるのはやめてください。"

    prompt = (
        f"あなたは{target_age}歳の子供向け絵本の作家です。子どもが分かるような簡単な言葉を使ってください。語り口調（ですます調）でお願いします。\n"
        f"以下の情報をもとに、{page_number}ページ目のストーリーを作成してください。\n"
        f"主役のキャラクター: {main_character} (名前: {main_character_name})\n"
        f"テーマ: {theme}\n"
        f"サブキャラクター: {', '.join(sub_characters)}\n"
        f"ストーリー構成: {storyline}\n"
        f"対象年齢: {target_age}歳\n"
        f"禁止ワード：「次のページ」、「最後のページ」、「…。」は使わないでください。\n"
        f"ページの内容は80文字程度（最大100文字）にしてください。\n\n"
        f"これまでのストーリー:\n{previous_content}\n\n"
        f"{page_number}ページ目のストーリー（日本語で簡潔に書いてください）: {ending_instruction}"
    )

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "あなたは日本語の幼児向け絵本作家です。やさしい語り口調で物語を話します。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# 画像生成プロンプト作成
def generate_image_prompt_from_story(story, main_character, theme, sub_characters):
    openai.api_key = OPENAI_API_KEY

    prompt = (
        f"You are an AI assistant specializing in creating illustration prompts for a consistent style children's book.\n"
        f"Based on the following story, craft a vivid, colorful, and child-friendly prompt for an illustration tool:\n\n"
        f"{story}\n\n"
        f"Include these details:\n"
        f"- The main character: {main_character}\n"
        f"- The theme: {theme}\n"
        f"- Sub-characters: {', '.join(sub_characters)}\n"
        f"Ensure the style remains consistent with the book's other illustrations, featuring vibrant colors and whimsical elements suitable for children aged 5."
    )

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You specialize in generating detailed illustration prompts for AI tools."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# 画像生成
def generate_image(prompt):
    headers = {
        "Api-Key": IDEOGRAM_API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    payload = {
        "image_request": {
            "prompt": prompt,
            "aspect_ratio": "ASPECT_1_1",
            "model": "V_2_TURBO",
            "style_type": "DESIGN",
            "negative_prompt": "text, watermark, logo, distorted features, unrelated elements"
        }
    }

    response = requests.post("https://api.ideogram.ai/generate", headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        if 'data' in data and data['data']:
            return data['data'][0]['url']
        else:
            print("Warning: No data found for the generated image.")
            return None
    else:
        print(f"Error: Failed to generate image. HTTP Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")
        return None

# ストーリーと画像の生成フロー
def generate_full_story_and_images(main_character, main_character_name, theme, sub_characters, storyline, target_age, num_pages):
    full_story = []
    image_urls = []

    for page_number in range(1, num_pages + 1):
        print(f"Generating story for page {page_number}...")
        page_story = generate_page_story(
            main_character=main_character,
            main_character_name=main_character_name,
            theme=theme,
            sub_characters=sub_characters,
            storyline=storyline,
            target_age=target_age,
            page_number=page_number,
            total_pages=num_pages,
            previous_content="\n".join(full_story)
        )
        full_story.append(page_story)

        print(f"Generating image prompt for page {page_number}...")
        image_prompt = generate_image_prompt_from_story(
            story=page_story,
            main_character=main_character_name,
            theme=theme,
            sub_characters=sub_characters
        )

        print("Generating image...")
        image_url = generate_image(image_prompt)
        image_urls.append(image_url)

    return full_story, image_urls

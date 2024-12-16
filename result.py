import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
import base64
from pathlib import Path

# Google Sheets APIの認証設定
def authenticate_google_sheets(json_keyfile):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile, scope)
    client = gspread.authorize(credentials)
    return client

# スプレッドシートにデータを保存
def save_to_google_sheets(spreadsheet_name, sheet_name, data, json_keyfile):
    client = authenticate_google_sheets(json_keyfile)
    sheet = client.open(spreadsheet_name).worksheet(sheet_name)
    sheet.append_row(data)  # 1行を追加

# スプレッドシートからデータを取得
def get_data_from_google_sheets(spreadsheet_name, sheet_name, json_keyfile):
    client = authenticate_google_sheets(json_keyfile)
    sheet = client.open(spreadsheet_name).worksheet(sheet_name)
    return sheet.get_all_records()  # 全データを取得

# 背景画像設定
background_image_path = Path(r"C:\Users\owner\STEP2-4\2024.12発表課題\ehonno_tane\product_image\Background.png")
logo_image_path = Path(r"C:\Users\owner\STEP2-4\2024.12発表課題\ehonno_tane\product_image\Logo.png")

def image_to_base64(image_path):
    return base64.b64encode(image_path.read_bytes()).decode()

background_base64 = image_to_base64(background_image_path)
logo_base64 = image_to_base64(logo_image_path)

# Streamlitでアウトプットを表示
def display_output(data):
    # 背景画像を設定
    background_style = f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{background_base64}");
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
            }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

    # ロゴの表示
    st.markdown(f"<img src='data:image/png;base64,{logo_base64}' style='width: 200px; display: block; margin: auto;'>", unsafe_allow_html=True)

    st.title("絵本生成結果")
    
    for row in data:
        title = row.get("Title", "タイトル不明")
        story_content = row.get("Story URL", "ストーリーがありません")
        image_url = row.get("Image URL", "")
        user_link = row.get("User Link", "#")

        # タイトルの表示
        st.header(f"タイトル: {title}")

        # 画像の表示
        if image_url:
            st.image(image_url, caption="生成された絵本の画像")

        # ストーリーの直接表示
        st.subheader("ストーリー")
        st.write(story_content)

        # ユーザーリンク
        st.markdown(f"**ユーザーリンク: [こちら]({user_link})**")

        # 区切り線
        st.markdown("---")

# メイン処理
if __name__ == "__main__":
    # 引数
    json_keyfile = r"C:\\Users\\owner\\STEP2-4\\2024.12発表課題\\try-kekka-6abb9855c399.json"
    spreadsheet_name = "EhonApp"
    sheet_name = "GeneratedBooks"

    # サンプルデータを定義して保存
    sample_data = ["Sample Title", "Once upon a time, there was a beautiful story...", "https://example.com/image.png", "https://example.com/userlink"]

    # データを保存
    save_to_google_sheets(spreadsheet_name, sheet_name, sample_data, json_keyfile)

    # スプレッドシートからデータを取得
    data = get_data_from_google_sheets(spreadsheet_name, sheet_name, json_keyfile)

    # データをStreamlitで表示
    if data:
        display_output(data)
    else:
        st.warning("データが見つかりませんでした。")

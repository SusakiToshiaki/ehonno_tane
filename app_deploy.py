import streamlit as st
import pandas as pd
import random
import base64
import re
from pathlib import Path
import zipfile
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import os
from story import generate_full_story_and_images
import time  # ロード中の遅延をシミュレート
from PIL import Image
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration
from google.cloud import vision
from deep_translator import GoogleTranslator
import spacy
import io
import gspread
from google.oauth2.service_account import Credentials

# ZIPファイルを解凍
model_path = "en_core_web_sm"
if not Path(model_path).exists():
    with zipfile.ZipFile("en_core_web_sm.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# ローカルディレクトリからモデルをロード
nlp = spacy.load(model_path)


# 必要な環境変数を取得
PRIVATE_KEY = st.secrets["google"]["PRIVATE_KEY"]
CLIENT_EMAIL = st.secrets["google"]["CLIENT_EMAIL"]
SPREADSHEET_ID = st.secrets["google"]["SPREADSHEET_ID"]
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]

# 環境変数のチェック
if not PRIVATE_KEY or not CLIENT_EMAIL or not SPREADSHEET_ID:
    raise EnvironmentError("環境変数 GOOGLE_PRIVATE_KEY, GOOGLE_CLIENT_EMAIL, または SPREADSHEET_ID が設定されていません。")

SERVICE_ACCOUNT_INFO = {
    "type": "service_account",
    "private_key": PRIVATE_KEY,
    "client_email": CLIENT_EMAIL,
    "token_uri": "https://oauth2.googleapis.com/token",
}

# Google Sheets API設定
RANGE_NAME = "DB!A:G"

# Ehon ID Automatic Generation Logic
def generate_next_book_id(worksheet):
    """
    スプレッドシート内の既存の絵本IDから最大値を取得し、次のIDを生成する
    """
    all_rows = worksheet.get_all_values()

    # Ehon-XXXXX形式のIDだけを抽出
    book_ids = [row[0] for row in all_rows if re.match(r"Ehon-\d{5}", row[0])]

    if not book_ids:
        # 初回のID生成時
        return "Ehon-00001"

    # 数字部分だけを抽出して最大値を計算
    max_id = max(int(book_id.split("-")[1]) for book_id in book_ids)
    next_id = f"Ehon-{max_id + 1:05d}"

    return next_id


# スプレッドシートからデータを取得する関数
def fetch_data_from_google_sheets():
    """
    Google Sheets APIを使用してスプレッドシートからデータを取得。
    サービスアカウントを使用して認証。
    """
    try:
        credentials = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)
        service = build("sheets", "v4", credentials=credentials)
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        rows = result.get("values", [])
        return rows[1:] if len(rows) > 1 else []  # ヘッダーを除外
    except Exception as e:
        st.error(f"スプレッドシートからデータを取得する際にエラーが発生しました: {str(e)}")
        return []

# ランダムなプロンプトを生成する関数
def generate_random_prompt(data):
    if not data:
        return "データが見つかりません。スプレッドシートを確認してください。"
    random_row = random.choice(data)
    maincharacter = random_row[0] if len(random_row) > 0 else ""
    maincharacter_name = random_row[1] if len(random_row) > 1 else ""
    location = random_row[2] if len(random_row) > 2 else ""
    theme = random_row[3] if len(random_row) > 3 else ""
    subcharacter_A = random_row[4] if len(random_row) > 4 else ""
    subcharacter_B = random_row[5] if len(random_row) > 5 else ""
    storyline = random_row[6] if len(random_row) > 6 else ""

    subcharacters = subcharacter_A
    if subcharacter_B:
        subcharacters += f", {subcharacter_B}"

    prompt_template = "{maincharacter}の{maincharacter_name}が、{location}を舞台に{subcharacters}たちと{theme}を学ぶ物語。ストーリは、{storyline}"
    return prompt_template.format(
        maincharacter=maincharacter,
        maincharacter_name=maincharacter_name,
        location=location,
        subcharacters=subcharacters,
        theme=theme,
        storyline=storyline
    )




# Step1 画像アップロード処理の関数
def upload_image():
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_container_width=True)
        st.success("画像が正常にアップロードされました！")
        return image
    
    return None

# Step2 画像解析に使う関数（3つ）　※画像の要素を抽出
# Step2−1 BLIPでキャプションを生成する間数
def generate_caption_blip(image): 

    # BLIPの準備
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # 画像を処理してキャプションを生成
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption
# Step2-2 VisionAIで画像のラベルを取得する関数（スコア0.8以上）
def extract_labels_visionai(image): 
    
    credentials = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)

    # Vision AIの準備
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # PIL画像をバイナリデータに変換
    image_byte_array = io.BytesIO()
    image.save(image_byte_array, format="PNG")
    content = image_byte_array.getvalue()

    # Vision AIリクエストの作成
    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    # スコアが0.8以上のラベルを抽出
    labels = [label.description for label in response.label_annotations if label.score >= 0.8]

    return labels
# Step2-3 キャプションから名詞のみを取得し、ラベルと結合。その後日本語へ翻訳する関数
def extract_nouns(caption, labels, target_language="ja"): 
    
    # spaCyモデルをロード
    nlp = spacy.load("en_core_web_sm")

    # キャプションを解析して名詞を抽出
    doc = nlp(caption)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # 名詞とラベルを結合し、重複を排除
    combined_list = list(set(nouns + labels))

    # 翻訳
    translator = GoogleTranslator(source="auto", target=target_language)
    translated_list = [translator.translate(word) for word in combined_list]

    return translated_list

# Step3 テーマを3つ生成する関数
def generate_themes(elements):

    # elements：Step2で抽出した画像の要素のこと
    prompt = f"""
    次の要素に基づいて、絵本のテーマを3つ提案してください:
    {", ".join(elements)}。

    条件:
    1. 各テーマはユニークであること。
    2. 子どもが興味を持てる楽しいテーマにすること。
    3. テーマのみ提案すること

    例:
    - 「自然と遊ぶ」
    - 「心をつなぐ笑顔」
    - 「アートで冒険」
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    # GPTの応答を整形してリスト化
    themes_text = response.choices[0].message.content.strip()
    themes = themes_text.split("\n")
    return [theme.strip("- ").strip() for theme in themes if theme.strip()]

# Step4 深掘り質問を生成する関数(画像要素と選択したテーマを基に生成する)
def generate_deep_questions(selected_theme, nouns):
    
    # nouns: Step2で抽出された画像の要素
    # selected_theme: Step3でユーザーが選択したテーマ
    prompt = f"""
    次の絵の要素に基づいて、物語のアイデアを深掘りする「問いかけ」を生成してください:
    {", ".join(nouns)}。
    テーマは「{selected_theme}」です。

    以下の条件を満たしてください:
    1. 各要素に1つの問いかけを提示する。
    2. 未就学児の子どもが答えやすく、想像力を広げられる形にする。
    3. 未就学児の子どもが考えた答えをもとに、さらに発展的なアイデアを引き出せる追加の問いかけを用意する。
    4. 問いかけのみ作成する。
    5. 作成する問いかけは１つ。

    例:
    - 雲: 「この雲は動いているみたい。どこに向かっているのかな？」→「その先にはどんな世界が広がっている？」
    - ネコ: 「このネコが話せるなら、何を教えてくれる？」→「教えてもらったことをどう使う？」
    - 木: 「この木のてっぺんに隠れたドアがあるみたい！どこに通じている？」→「そのドアを開けると、どんな冒険が始まる？」

    # 1つの問いかけを生成してください。
    """
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは創造的な絵本のアイデアを生成するプロフェッショナルです。"},
            {"role": "user", "content": prompt}
        ]
    )

    # ChatCompletion の戻り値を正しく参照
    questions_text = response.choices[0].message.content.strip()
    questions = questions_text.split("\n")
    return [q.strip("- ").strip() for q in questions if q.strip()]

# Step5 絵本生成に必要な情報を作成する関数
def story_elements(selected_theme, nouns, questions, user_answers):
    prompt = f"""
    次の要素とユーザーからの情報に基づいて、絵本の生成に必要な情報を生成してください:
    {", ".join(nouns)}。
    テーマは「{selected_theme}」です。

    事前に問いかけした内容:
    {questions}

    ユーザーからの情報:
    {user_answers}

    以下の条件を満たしてください:
    1. 以下の構造で情報を生成してください:
       - maincharacter: 主人公の説明
       - maincharacter_name: 主人公の名前
       - location: 舞台となる場所
       - theme: 絵本のテーマ
       - subcharacter_A: サブキャラクターAの説明
       - subcharacter_B: サブキャラクターBの説明
       - storyline: 絵本のストーリーライン
    2. 空欄の場合は「未設定」と記載してください。
    3. 他の項目の説明が十分に詳細であること。

    例:
    maincharacter: 村の祭りに参加する陶芸家
    maincharacter_name: あや
    location: イタリアの丘陵地帯
    theme: 伝統と芸術
    subcharacter_A: ジュリア（陶器職人の少女）
    subcharacter_B: ルカ（祭りの企画者）
    storyline: あやは陶芸の技術を学ぶため訪れた村で、祭りを通じて地元の人々と交流し、芸術の中に隠された物語を知る。
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは創造的な絵本のアイデアを生成するプロフェッショナルです。"},
            {"role": "user", "content": prompt}
        ]
    )

    # ChatCompletion の戻り値を正しく参照
    elements_text = response.choices[0].message.content.strip()
    elements = elements_text.split("\n")
    
    # 辞書形式に変換
    story_dict = {}
    for element in elements:
        if ": " in element:  # 「キー: 値」の形式で分割
            key, value = element.split(": ", 1)
            story_dict[key.strip()] = value.strip()
    
    return story_dict

# Step6 生成された絵本情報をスプレッドシートに追記する関数
def append_story_elements_to_sheet(story_elements, worksheet):
    # story_elements: Step5で生成された絵本情報（辞書型）

    # スプレッドシートに追記するための行データを生成
    new_row = [
        story_elements.get("maincharacter", "未設定"),
        story_elements.get("maincharacter_name", "未設定"),
        story_elements.get("location", "未設定"),
        story_elements.get("theme", "未設定"),
        story_elements.get("subcharacter_A", "未設定"),
        story_elements.get("subcharacter_B", "未設定"),
        story_elements.get("storyline", "未設定")
    ]

    # データをスプレッドシートに追記
    worksheet.append_row(new_row, value_input_option="USER_ENTERED") 


# 背景画像設定
background_image_path = Path(r"C:\Users\toshi\ehonnotane\ehonno_tane\product_image\Background.png")
logo_image_path = Path(r"C:\Users\toshi\ehonnotane\ehonno_tane\product_image\Logo.png")

def image_to_base64(image_path):
    return base64.b64encode(image_path.read_bytes()).decode()

background_base64 = image_to_base64(background_image_path)
logo_base64 = image_to_base64(logo_image_path)

# ページ状態の初期化
if "page" not in st.session_state:
    st.session_state.page = "main"

# ページ切り替え関数
def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# CSSを適用するMarkdown
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    header {{
        visibility: hidden;
        height: 0;
    }}
    
    .logo-container {{
        text-align: center;
        margin-top: 20px;
    }}

    .logo-container img {{
        max-width: 50%;
        height: auto;
        display: inline-block;
    }}
    .stButton > button {{
        background-color: #59008B;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: #A122EA;
        color: white;
    }}

    /* ラジオボタン全体（ラベル部分）のスタイル */
    [data-baseweb="radio"] {{
        display: flex;
        align-items: center;
        margin-bottom: 20px; /* 各ラジオボタンの間隔を広げる */
        padding: 15px 20px; /* 内側の余白を調整 (上下15px, 左右20px) */
    }}

    /* チェック済みのラジオボタンの背景 */
    [data-baseweb="radio"]:has(input:checked) {{
        background-color: rgba(255, 255, 255, 0.1); /* 半透明の背景色 */
        border-radius: 8px; /* 角を丸くする */
        padding: 15px 20px; /* 内側の余白を調整 (上下15px, 左右20px) */
    }}

    /* 丸いラジオボタン（選択部分）のデザイン */
    [data-baseweb="radio"] div.st-bh {{
        border: 2px solid white !important; /* ラジオボタンの枠線を白に */
        border-radius: 50%; /* 完全な丸にする */
    }}

    /* ラベル内のテキスト（説明部分）のスタイル */
    [data-baseweb="radio"] div.st-bc p {{
        color: white !important; /* テキストの色を白に変更 */
        font-size: 16px; /* フォントサイズを調整 */
        margin-left: 10px; /* ボタンとテキスト間の余白を調整 */
    }}

    /* ボタンのカスタマイズ */
    .stButton > button {{
        background-color: #59008B;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }}

    /* ボタンのホバースタイル */
    .stButton > button:hover {{
        background-color: #A122EA;
        color: white;
    }}
    

    
    </style>
    """,
    unsafe_allow_html=True
)

# メインページ
if st.session_state.page == "main":
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" alt="ロゴ">
        </div>
        <div class="center-content": margin-bottom: 50px>
            <h1>毎日少しずつ進む、親子だけの冒険絵本 🪄</h1>
            <p>このアプリは、親子で物語の展開を予想しながら楽しむ、3日間限定の特別な絵本です。<br>
            一度に読み進められるのは少しずつでも、その分「明日はどうなるの？」とドキドキが続きます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("おまかせしちゃう"):
        set_page("A")
    if st.button("オリジナルの物語を作りたい！"):
        set_page("B")

    # 絵本ID入力欄を追加
    st.write("または、絵本IDを入力して保存された絵本データを呼び出してください:")
    input_book_id = st.text_input("絵本IDを入力", "Ehon-", key="book_id_input")
    
    if st.button("絵本を表示"):
        if input_book_id:
            # スプレッドシートから絵本データを取得
            SCOPES = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            credentials = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
            client = gspread.authorize(credentials)
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            
            try:
                worksheet = spreadsheet.worksheet("GeneratedBooks")
                rows = worksheet.get_all_values()  # 全データを取得
                
                # 入力された絵本IDに対応するデータを検索
                book_data = [row for row in rows if row[0] == input_book_id]
                
                if book_data:
                    # 該当データが見つかった場合、セッションに保存してResultページへ
                    st.session_state["loaded_book_data"] = book_data
                    set_page("result")
                else:
                    st.warning("指定された絵本IDが見つかりませんでした。")
            except Exception as e:
                st.error(f"スプレッドシートのデータ取得中にエラーが発生しました: {e}")
        else:
            st.warning("絵本IDを入力してください。")


#############かえページ
elif st.session_state.page == "A":
    if "data" not in st.session_state:
        st.session_state.data = fetch_data_from_google_sheets()

    if "prompts" not in st.session_state:
        st.session_state.prompts = [generate_random_prompt(st.session_state.data) for _ in range(3)]

    prompts = st.session_state.prompts
    st.markdown('<h1 style="color: white; text-align: center;">どの物語を読む？</h1>', unsafe_allow_html=True)

    
    selected_prompt = st.radio("", prompts)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("メインページへ戻る", key="back_to_main"):
            set_page("main")  # セッションステートを更新してメインページへ戻る
    with col2:
        if st.button("次へ"):
            if selected_prompt:
                st.session_state["selected_prompt"] = selected_prompt
            set_page("result")  # 次のページ（Resultページ）に遷移



#############えんちゃんページ
elif st.session_state.page == "B":

    st.title("描いた絵が、すてきな絵本に！")
    st.title("") # 余白用
    st.subheader("使い方はとっても簡単！")
    st.subheader("") # 余白用
    st.write("Step1　描いた絵をアップロード")
    st.write("Step2　絵にまつわるテーマを選択")
    st.write("Step3　簡単な質問に答えるだけ")
    st.subheader("") # 余白用

    # さっそく作ってみるボタン
    if st.button("さっそく作ってみる", key="B_Step1"):
        set_page("B_Step1")

    # 「メインページへ戻る」ボタン
    if st.button("メインページへ戻る", key="result_back_to_main"):
        set_page("main")  # メインページに遷移

elif st.session_state.page == "B_Step1":

    st.title("Step１　描いた絵をアップロード！")
    st.title("") # 余白用
    st.title("") # 余白用

    # 関数呼び出し
    uploaded_image = upload_image()

    # 画像アップロード後
    if uploaded_image:
        # セッションにアップロードした画像を保存
        st.session_state.uploaded_image = uploaded_image

        # 列を作成（左側を広くして右端にボタンを配置）
        # 次に進むボタン
        if st.button("次のステップへ進む", key="B_Step2"):
            set_page("B_Step2")

        # 「メインページへ戻る」ボタン
        if st.button("メインページへ戻る", key="result_back_to_main"):
            set_page("main")  # メインページに遷移

elif st.session_state.page == "B_Step2":
    
    st.title("Step２　絵本のテーマを選択！")
    st.title("") # 余白用
    st.title("") # 余白用

    # アップロードされた画像を取得
    uploaded_image = st.session_state.get("uploaded_image")

    # 画像解析が未実行の場合のみ実行
    if "is_image_analyzed" not in st.session_state or not st.session_state.is_image_analyzed:

        # セッションから画像を取得
        uploaded_image = st.session_state.uploaded_image

        # BLIPによるキャプション生成
        with st.spinner("BLIPでキャプションを生成中..."):
            caption = generate_caption_blip(uploaded_image)
            st.session_state.caption = caption  # キャプションをセッションに保存

        # Vision AIによるラベル抽出
        with st.spinner("Vision AIでラベルを抽出中..."):
            labels = extract_labels_visionai(uploaded_image)
            st.session_state.labels = labels  # ラベルをセッションに保存
        
        # 名詞抽出処理を呼び出し
        with st.spinner("キャプションから名詞を抽出中..."):
            nouns = extract_nouns(caption, labels, target_language="ja")  # 外部関数を使用
            st.session_state["nouns"] = nouns  # セッションに保存
        
        # 解析済みフラグをTrueに設定
        st.session_state.is_image_analyzed = True
        st.success("画像解析が完了しました！")
    
    # 解析結果をセッションから取得
    caption = st.session_state.get("caption", "")
    labels = st.session_state.get("labels", [])
    nouns = st.session_state.get("nouns", [])

    # テーマ生成済みか確認し、未生成の場合のみ実行
    if "themes" not in st.session_state:
        with st.spinner("テーマを生成中..."):
            themes = generate_themes(nouns)
            st.session_state["themes"] = themes
            st.success("テーマが生成されました！")
        
    # レイアウト: 左に画像、右に解析結果
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(uploaded_image, caption="アップロードされた画像", use_container_width=True)

    with col2:
        # st.subheader("絵本のテーマを選択してね")

        

        # テーマを表示して選択
        themes = st.session_state.get("themes", [])
        if themes:
            # 初期値をセッション状態に設定
            if "selected_theme" not in st.session_state:
                st.session_state["selected_theme"] = themes[0]  # 最初のテーマをデフォルトに設定

            st.subheader("以下のテーマから1つ選んでね！")
            # ラジオボタンでテーマを選択
            selected_theme = st.radio(
                "以下のテーマから1つ選んでください：", 
                themes, 
                index=themes.index(st.session_state["selected_theme"])
            )

            # 選択されたテーマをセッション状態に保存
            st.session_state["selected_theme"] = selected_theme

            # 選択したテーマを表示
            st.write(f"選択されたテーマ: **{st.session_state['selected_theme']}**")
        else:
            st.error("テーマ生成に失敗しました。もう一度お試しください。")
        
        # 列を作成（左側を広くして右端にボタンを配置）
        # 次に進むボタン
        if st.button("次のステップへ進む", key="B_Step3"):
            set_page("B_Step3")

        # 「メインページへ戻る」ボタン
        if st.button("Step１へ戻る", key="B_Step1"):
            set_page("B_Step1") 

elif st.session_state.page == "B_Step3":
    st.title("Step３　絵についてもっと教えて！")
    st.title("") # 余白用
    st.title("") # 余白用

    # アップロードされた画像を取得
    uploaded_image = st.session_state.get("uploaded_image")
        
    # レイアウト: 左に画像、右に解析結果
    col1, col2 = st.columns([1,2])

    with col1:
        st.image(uploaded_image, caption="アップロードされた画像", use_container_width=True)

    with col2:
        
        # セッションから必要な情報を取得
        selected_theme = st.session_state.get("selected_theme", "")
        nouns = st.session_state.get("nouns", [])
        
        # 質問をセッション状態に保存
        if selected_theme and nouns:
            # 質問がまだ生成されていない場合に処理を実行
            if "deep_questions" not in st.session_state:
                with st.spinner("絵に関する質問を生成中..."):
                    try:
                        questions = generate_deep_questions(selected_theme, nouns)
                        st.session_state["deep_questions"] = questions  # セッション状態に保存
                        st.success("絵に関する質問が生成されました！")
                    except Exception as e:
                        st.error(f"絵に関する質問の生成に失敗しました: {e}")

        # 質問に対するユーザーの回答を保存
        if "deep_questions" in st.session_state:
            questions = st.session_state["deep_questions"]

            # 回答をセッションで管理
            if "user_answers" not in st.session_state:
                st.session_state["user_answers"] = {}

            # 各質問に対して回答を入力
            st.write("### あなたの回答を入力してください")
            for i, question in enumerate(questions):
                if "→" in question:
                    main_question, follow_up = question.split("→", 1)
                else:
                    main_question, follow_up = question, None

                # メイン質問の回答
                main_answer = st.text_input(main_question.strip(), key=f"main_{i}")
                follow_up_answer = (
                    st.text_input(f"深掘り質問: {follow_up.strip()}", key=f"follow_up_{i}")
                    if follow_up
                    else "なし"
                )

                # セッションに回答を保存
                st.session_state["user_answers"][main_question.strip()] = {
                    "main": main_answer,
                    "follow_up": follow_up_answer,
                }

    # 列を作成（左側を広くして右端にボタンを配置）
        # 次に進むボタン
        if st.button("絵本を生成する", key="generative_ehon"):
            # 絵本情報を生成
            with st.spinner("生成中..."):
                try:
                    # セッションから回答を取得
                    user_answers = st.session_state.get("user_answers", {})
                    story_elements = story_elements(selected_theme, nouns, questions, user_answers)
                    st.session_state["story_elements"] = story_elements

                    # Google Sheets API 設定
                    SCOPES = [
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                    credentials = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
                    client = gspread.authorize(credentials)
                    spreadsheet = client.open_by_key(SPREADSHEET_ID)
                    worksheet = spreadsheet.sheet1

                    # Step6 絵本情報をスプレッドシートに追記
                    # 絵本情報をスプレッドシートに追記
                    try:
                        append_story_elements_to_sheet(story_elements, worksheet)
                        st.success("スプレッドシートに追記しました")
                    except Exception as e:
                        st.error(f"スプレッドシートへの追記に失敗しました: {e}")

                    # 結果を表示
                    st.success("生成が完了しました！")

                    #Debug
                    st.write("### 生成された絵本情報")
                    st.write(story_elements)

                    #To Result
                    st.session_state.page ="result"
                    st.rerun()
                
                except Exception as e:
                    st.error(f"生成中にエラーが発生しました: {e}")
                
        # 「メインページへ戻る」ボタン
        if st.button("Step２へ戻る", key="B_Step2"):
            set_page("B_Step2") 


############# Resultページ
elif st.session_state.page == "result":

    # ロゴをページの上部に表示
    logo_path = Path("product_image/Logo.png")  # ロゴ画像のパス
    logo_base64 = image_to_base64(logo_path)  # Base64エンコード

    st.markdown(
        f'''
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 200px; margin-bottom: 20px;">
        </div>
        ''',
        unsafe_allow_html=True
    )

    st.title("📚 絵本を表示 📚")

    # 保存済みの絵本データがセッションにある場合
    if "loaded_book_data" in st.session_state:
        # スプレッドシートからロードされたデータを使用
        book_data = st.session_state["loaded_book_data"]
        book_id = book_data[0][0]  # 絵本IDを取得

        st.success(f"保存された絵本データを表示しています！ 絵本ID: **{book_id}**")
        st.markdown('<h2 style="text-align: center;">📖 あなたの絵本 📖</h2>', unsafe_allow_html=True)

        for row in book_data:
            page_number = row[1]
            story = row[2]
            image_url = row[3]

            st.markdown(f"### ページ {page_number}")
            st.write(story)
            if image_url:
                st.image(image_url, caption=f"ページ {page_number} のイラスト")
            else:
                st.warning(f"ページ {page_number} の画像が見つかりません。")

    # 新しい絵本を生成する場合
    elif "selected_prompt" in st.session_state or "story_elements" in st.session_state:
        st.write("新しい絵本を生成")

        # (A) 選択されたプロンプトがある場合
        if "selected_prompt" in st.session_state and st.session_state["selected_prompt"]:
            selected_prompt = st.session_state["selected_prompt"]
            random_row = selected_prompt.split(", ")

            # プロンプトからストーリー要素を取得
            maincharacter, maincharacter_name, location, theme, subcharacter_A, subcharacter_B, storyline = (
                random_row + [""] * 7
            )[:7]

        # (B) 画像解析から生成されたストーリー要素がある場合
        elif "story_elements" in st.session_state and st.session_state["story_elements"]:
            story_elements = st.session_state["story_elements"]

            # 画像解析結果からストーリー要素を取得
            maincharacter = story_elements.get("maincharacter", "")
            maincharacter_name = story_elements.get("maincharacter_name", "")
            location = story_elements.get("location", "")
            theme = story_elements.get("theme", "")
            subcharacter_A = story_elements.get("subcharacter_A", "")
            subcharacter_B = story_elements.get("subcharacter_B", "")
            storyline = story_elements.get("storyline", "")

        else:
            # データがどちらにも存在しない場合
            st.error("プロンプトまたは画像解析結果が見つかりません。")
            st.stop()

        # サブキャラクターをリストにまとめる
        sub_characters = [char for char in [subcharacter_A, subcharacter_B] if char]

        # 絵本を生成
        with st.spinner("絵本を生成しています。少々お待ちください..."):
            try:
                full_story, image_urls = generate_full_story_and_images(
                    main_character=maincharacter,
                    main_character_name=maincharacter_name,
                    theme=theme,
                    sub_characters=sub_characters,
                    storyline=storyline,
                    target_age=5,
                    num_pages=5,
                )

                # Google Spreadsheetへの保存準備
                SCOPES = [
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]

                credentials = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=SCOPES)
                client = gspread.authorize(credentials)
                spreadsheet = client.open_by_key(SPREADSHEET_ID)

                # "GeneratedBooks"タブを取得または作成
                try:
                    worksheet = spreadsheet.worksheet("GeneratedBooks")
                except gspread.exceptions.WorksheetNotFound:
                    worksheet = spreadsheet.add_worksheet(title="GeneratedBooks", rows=1000, cols=10)
                    worksheet.append_row(["絵本ID", "ページ番号", "ページの話", "IdeogramのURL"])

                # 絵本IDを自動生成
                book_id = generate_next_book_id(worksheet)

                # データをスプレッドシートに保存
                for page_number, (story, image_url) in enumerate(zip(full_story, image_urls), 1):
                    worksheet.append_row([
                        book_id,                # 絵本ID
                        page_number,            # ページ番号
                        story,                  # ページの話
                        image_url               # IdeogramのURL
                    ])

            except Exception as e:
                st.error(f"絵本の生成中にエラーが発生しました: {e}")
                st.stop()

        # 結果を表示
        st.success(f"絵本が完成しました！ あなたの絵本IDは **{book_id}** です！")
        st.markdown("この絵本IDを保存しておけば、後で絵本を再表示することができます！")
        st.markdown('<h2 style="text-align: center;">📖 あなたの絵本 📖</h2>', unsafe_allow_html=True)

        for i, (story, image_url) in enumerate(zip(full_story, image_urls), 1):
            st.markdown(f"### ページ {i}")
            st.write(story)
            if image_url:
                st.image(image_url, caption=f"ページ {i} のイラスト")
            else:
                st.warning(f"ページ {i} の画像生成に失敗しました。")

    else:
        st.error("絵本データが見つかりません。メインページに戻り、絵本IDを入力するか、新しい絵本を作成してください。")
        st.stop()

    # メインページに戻るボタン
    if st.button("メインページへ戻る"):
        set_page("main")

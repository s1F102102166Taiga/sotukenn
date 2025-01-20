from flask import Flask, request, render_template, jsonify,session
import lyricsgenius
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cosine
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

app = Flask(__name__)

app.secret_key = 'fnakldmnaobnalmgkljabdklmgaadagjfhauihbijnvaopbuina'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="7e92175d0e2e4611bd34ee7b6d190199",
    client_secret="972dc58850ae4acb958d1935dc0e4250"
))


# Genius APIの設定
genius_api_key='QZwsfuioPzykjvq-a-wnDq4wzfUwFpaIaC6sqRdUduwLvOZXZI7QvW44f3-3lR1X'
genius = lyricsgenius.Genius(
    genius_api_key,
    timeout=20  # タイムアウトを15秒に延長
)

# モデルの読み込み
model_dir = './sotuken_model'  # ローカルディレクトリのパス

model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,  # ローカルのモデルディレクトリ
    use_safetensors=True,      # safetensorsを使う場合
    local_files_only=True      # ローカルファイルのみを使う設定
)
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


emotion_groups = {
    "楽しい": ["喜び", "期待", "信頼"],
    "fun": ["喜び", "期待", "信頼"],  # 英語名を追加
    "悲しい": ["悲しみ", "嫌悪"],
    "sad": ["悲しみ", "嫌悪"],  # 英語名を追加
    "怖い": ["怒り", "恐れ", "驚き"],
    "scary": ["怒り", "恐れ", "驚き"]  # 英語名を追加
}
emotion_colors = {
    "喜び": "#FFD700",  # 黄色
    "悲しみ": "#00BFFF",  # 青
    "期待": "#32CD32",  # 緑
    "驚き": "#FF6347",  # 赤
    "怒り": "#FF4500",  # オレンジ
    "恐れ": "#A9A9A9",  # グレー
    "嫌悪": "#8B0000",  # ダークレッド
    "信頼": "#8A2BE2"   # 紫
}

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def generate_emotion_barchart(emotion_scores, song_title=None):
    """
    感情スコアの棒グラフを生成し、画像データを返す。
    """
    plt.figure(figsize=(8, 3))
    df = pd.DataFrame(emotion_scores.items(), columns=['感情', 'スコア'])
    sns.barplot(x='感情', y='スコア', data=df, palette='pastel')
    plt.title(f"曲: {song_title}" if song_title else "感情分析結果", fontsize=15)
    plt.ylabel("スコア")
    plt.xlabel("感情")
    plt.xticks(rotation=45)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


from bs4 import BeautifulSoup
import requests



def get_song_url(title, artist_name, genius_api_key, retries=1, delay=5):
    """Genius APIで曲のURLを取得する。失敗時はリトライを試みる"""
    for attempt in range(retries):
        try:
            
            # Genius APIで曲を検索（直接リクエストを使用）
            song_url = search_song_directly(title, artist_name, genius_api_key)
        
            if song_url:
                return song_url
            else:
                print(f"Song URL not found for '{title}' by '{artist_name}'.")

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Error fetching song URL for Title: {title}, Artist: {artist_name}: {e}")
    
    print(f"Failed to fetch song URL for Title: {title}, Artist: {artist_name} after {retries} attempts")
    return None


def search_song_directly(title, artist_name, genius_api_key):
    """Genius APIに直接リクエストを送り、曲情報を取得"""
    base_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {genius_api_key}"}
    params = {"q": f"{title} {artist_name}"}
    
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        response_data = response.json()
        hits = response_data.get("response", {}).get("hits", [])
        if hits:
            for hit in hits:
                result = hit.get("result", {})
                song_url = result.get("url")
                if song_url:
                    print(f"Found song URL: {song_url}")
                    return song_url  # 最初の結果のURLを返す
        else:
            print("No hits found in Genius API response.")
    else:
        print(f"Failed to fetch data from Genius API. Status code: {response.status_code}")
    
    return None


def normalize_text(text):
    """曲名やアーティスト名を正規化"""
    import re
    return re.sub(r'\s+', ' ', text).strip()  # 不要な空白を削除


# 感情分析用の関数
def analyze_emotion(text, show_fig=False, ret_prob=False):
    try:
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        preds = model(**tokens)
        probabilities = np.exp(preds.logits.detach().numpy()) / np.sum(np.exp(preds.logits.detach().numpy()))
        emotions = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"]
        
        emotion_scores = dict(zip(emotions, probabilities[0]))
        
        # デバッグ用に感情スコアを表示
        print(f"Emotion Scores: {emotion_scores}")
        
        if ret_prob:
            return emotion_scores
        else:
            return emotion_scores

    except Exception as e:
        print(f"Error processing emotion analysis: {e}")
        return {"error": str(e)}

def convert_to_float(value):
    if isinstance(value, np.float32):
        return float(value)
    elif isinstance(value, dict):
        return {key: convert_to_float(val) for key, val in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(item) for item in value]
    else:
        return value
    
import time

def fetch_lyrics_from_url(url):
    try:
        print(f"Fetching page content for URL: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Page successfully fetched for URL: {url}")
            soup = BeautifulSoup(response.text, 'html.parser')

            # 最新のGeniusの歌詞セクションを取得
            lyrics_divs = soup.find_all('div', {'data-lyrics-container': 'true'})
            if lyrics_divs:
                lyrics = "\n".join([div.get_text(strip=True) for div in lyrics_divs])
                print(f"Lyrics extracted: {lyrics[:100]}...")  # 抜粋を表示
                return lyrics
            else:
                print("Lyrics section not found in the page.")
        else:
            print(f"Failed to fetch page. Status code: {response.status_code}")
        return None
    except Exception as e:
        print(f"Error fetching lyrics from URL: {e}")
        return None



def search_song_directly(title, artist_name, genius_api_key):
    base_url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {genius_api_key}"}
    params = {"q": f"{title} {artist_name}"}
    
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        response_data = response.json()
        print(f"DEBUG: Response data: ")  # デバッグ用にレスポンスを表示
        hits = response_data.get("response", {}).get("hits", [])
        if hits:
            for hit in hits:
                result = hit.get("result", {})
                song_url = result.get("url")
                if song_url:
                    print(f"Found song URL: {song_url}")
                    return song_url  # 最初の結果のURLを返す
        else:
            print("No hits found in Genius API response.")
    else:
        print(f"Failed to fetch data from Genius API. Status code: {response.status_code}")
    
    return None






def split_song_artist(song):
    if isinstance(song, dict):  # songが辞書型の場合
        title = song.get("song", "").strip()  # 曲名を取得
        artist = song.get("artist", "").strip()  # アーティスト名を取得
        return title, artist
    elif isinstance(song, str):  # songが文字列型の場合
        if '（作曲者:' in song:
            title, artist = song.split('（作曲者:')
            return title.strip(), artist.replace('）', '').strip()
        return song.strip(), None
    else:
        raise ValueError(f"Unsupported data type for song: {type(song)}")

def format_song_data(song, artist):
    if artist:
        return f"{song}（作曲者: {artist}）"
    else:
        return f"{song}（作曲者: 不明）"



def get_song_lyrics(title, artist_name):
    try:
        # タイトルとアーティスト名の存在を確認
        if not title or not isinstance(title, str):
            print(f"Invalid title: {title}")
            return None
        
        # Genius APIで歌詞を検索
        song = genius.search_song(title, artist=artist_name)
        
        # APIレスポンスを確認（デバッグ用）
        if song:
            print(f"API Response for {title}: {song.to_dict()}")  # Genius APIレスポンス全体を表示
            if song.lyrics:
                clean_lyrics = "\n".join(song.lyrics.split("\n"))  # 不要な行削除
                return clean_lyrics

        # URLから直接歌詞を取得する試み
        if song and song.url:
            print(f"Attempting to fetch lyrics from URL: {song.url}")
            lyrics_from_url = get_song_url(song.url)
            if lyrics_from_url:
                print(f"Lyrics successfully fetched from URL: {song.url}")
                return lyrics_from_url
            else:
                print(f"Lyrics could not be fetched from URL: {song.url}")
        else:
            print("No URL available to fetch lyrics.")

        # 歌詞が見つからなかった場合
        print(f"Lyrics not found for {title} by {artist_name}")
        return None

    except requests.exceptions.Timeout:
        print(f"Timeout occurred while fetching lyrics for {title} by {artist_name}")
        return None
    except Exception as e:
        print(f"Error fetching lyrics for {title} by {artist_name}: {e}")
        return None


def recommend_songs_by_emotion(group_name, selected_songs):
    group_emotions = emotion_groups[group_name]
    
    aggregated_scores = {emotion: 0 for emotion in group_emotions}
    for song in selected_songs:
        scores = song.get("scores", {})
        for emotion in group_emotions:
            aggregated_scores[emotion] += scores.get(emotion, 0)
    
    numeric_columns = dataset.select_dtypes(include=[float, int]).columns
    emotion_columns = [col for col in group_emotions if col in numeric_columns]

    dataset['similarity'] = dataset.apply(
        lambda row: 1 - cosine(
            [aggregated_scores.get(emotion, 0) for emotion in emotion_columns],
            [row[emotion] for emotion in emotion_columns]
        ) if all(emotion in row for emotion in emotion_columns) else 0,
        axis=1
    )

    top_matches = dataset.nlargest(5, 'similarity')
    
    recommendations = []
    for _, match in top_matches.iterrows():
        scores = {emotion: match[emotion] for emotion in numeric_columns}
        recommendations.append({
            "name": match['track_name'],
            "formatted": f"{match['track_name']}（作曲者: {match['artists']}）",
            "scores": scores,
            "graph": generate_emotion_barchart(scores, song_title=match['track_name'])
        })
    return recommendations



# 初期ページ（ques.htmlを表示）
@app.route('/')
def index():
    return render_template('ques.html')

dataset = pd.read_csv("A_dataset.csv")

@app.route('/ques', methods=['POST'])
def ques():
    # フォームから取得したJSONデータをパース
    try:
        selected_songs = request.form['songs']
        # JSON文字列の場合は辞書型に変換
        if isinstance(selected_songs, str):
            import json
            selected_songs = json.loads(selected_songs)
    except Exception as e:
        print(f"Error parsing input data: {e}")
        selected_songs = []
    all_emotions = {}
    hidden_parameters = {}
    recommended_songs = []
    graphs = {}
    selected_songs_list = []
    past_recommendations = session.get('past_ques_recommendations', [])

    # 曲ごとの処理
    for song_data in selected_songs:
        print(f"Processing song_data: {song_data}")
        title, artist = split_song_artist(song_data)
        title = normalize_text(title)
        artist = normalize_text(artist)
        print(f"DEBUG: Title: {title}, Artist: {artist}")

        response_data = search_song_directly(title, artist, genius_api_key)
        if response_data:
            print("DEBUG: Raw API Response:")
        else:
            print("No response data found.")
        if not title:
            print(f"Skipping invalid song: {song_data}")
            continue  # タイトルがない場合はスキップ

    # 曲名をキーとして使用
        song_url = get_song_url(title, artist,genius_api_key)
        if song_url:
            print(f"Song URL obtained:")
            # URLから歌詞を取得
            song_lyrics = fetch_lyrics_from_url(song_url)
            if song_lyrics:
                # 感情分析を実行
                print("Lyrics retrieved successfully:\n", song_lyrics[:100])
                emotion_scores = analyze_emotion(song_lyrics, ret_prob=True)
                all_emotions[title] = emotion_scores
                hidden_parameters[title] = {"scores": emotion_scores}
                print(f"Emotion scores for {title}: {emotion_scores}")
                graphs[title] = generate_emotion_barchart(emotion_scores, song_title=title)
                formatted_title = format_song_data(title, artist)
                selected_songs_list.append({
                    "name": title,
                    "formatted": formatted_title,
                    "scores": emotion_scores
                })
            else:
                print(f"Failed to fetch lyrics from URL: ")
                all_emotions[title] = {"error": "歌詞取得エラー"}
        else:
            print(f"No URL found for song: {title}")
            all_emotions[title] = {"error": "URL取得エラー"}


    if all_emotions:
        aggregated_emotions = pd.DataFrame.from_dict(all_emotions, orient='index').mean().to_dict()
        print(f"Aggregated emotion scores: {aggregated_emotions}")
    else:
        return render_template('home.html', selected_songs=[], recommendations=[])
    # 推薦アルゴリズム：数値型の列のみを対象にする
    # 推薦アルゴリズム：数値型の列のみを対象にする
    numeric_columns = dataset.select_dtypes(include=[float, int]).columns
    emotion_columns = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
    numeric_columns = [col for col in emotion_columns if col in dataset.columns]

    if not numeric_columns:
        print("No valid emotion columns found in the dataset.")
        return render_template('home.html', selected_songs=[], recommendations=[])

    dataset[numeric_columns] = dataset[numeric_columns].fillna(0)

    # 類似度計算
    similarities = []
    for index, row in dataset.iterrows():
        try:
            survey_vector = [aggregated_emotions.get(emotion, 0) for emotion in numeric_columns]
            dataset_vector = row[numeric_columns].values.astype(float)

            if np.linalg.norm(survey_vector) == 0 or np.linalg.norm(dataset_vector) == 0:
                similarities.append(0)
                continue

            similarity = 1 - cosine(survey_vector, dataset_vector)
            similarities.append(similarity)
        except Exception as e:
            print(f"Error calculating similarity for row {index}: {e}")
            similarities.append(0)

    # データセットに類似度を追加
    dataset['similarity'] = similarities
    print(f"Dataset with similarity scores:\n{dataset[['track_name', 'similarity']].head()}")

    # 上位5曲を取得
    top_matches = dataset.nlargest(100, 'similarity')  # 多めに取得してフィルタ

    # 過去のおすすめを除外
    new_recommendations = [
        match for _, match in top_matches.iterrows()
        if match['track_name'] not in past_recommendations
    ][:5]  # 新しい曲を5件取得

    if new_recommendations:
        session['past_ques_recommendations'] = past_recommendations + [
            match['track_name'] for match in new_recommendations
        ]

    for match in new_recommendations:
        scores = {emotion: match[emotion] for emotion in numeric_columns}
        recommended_songs.append({
            "name": match['track_name'],
            "formatted": f"{match['track_name']}（作曲者: {match['artists']}）",
            "scores": scores
        })
        print(f"Recommended song: {match['track_name']} with similarity {match['similarity']:.4f}")

    return render_template(
        'home.html',
        selected_songs=selected_songs_list,
        recommendations=recommended_songs,
        hidden_params={**hidden_parameters, **{song['name']: {'scores': song['scores']} for song in recommended_songs}},
        emotion_colors=emotion_colors,
        graphs=graphs
    )




@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/search_artist', methods=['GET'])
def search_artist():
    query = request.args.get('query')
    if query:
        results = sp.search(q=query, type='artist', limit=5)  # Spotify APIで上位5件のアーティストを取得
        artists = [{"name": artist['name'], "id": artist['id']} for artist in results['artists']['items']]
        return jsonify(artists)
    return jsonify([])

@app.route('/search_song', methods=['GET'])
def search_song():
    query = request.args.get('query')
    if query:
        results = sp.search(q=query, type='track', limit=5)  # Spotify APIで上位5件の曲を取得
        songs = [
            {"name": song['name'], "artist": song['artists'][0]['name'], "id": song['id']} 
            for song in results['tracks']['items']
        ]
        return jsonify(songs)
    return jsonify([])

@app.route('/<string:mood>', methods=['GET'])
def recommend_songs(mood):
    # 感情グループに基づいて対象感情を取得
    if mood not in emotion_groups:
        return "Invalid mood", 404

    target_emotions = emotion_groups[mood]

    # 感情スコア列のみを対象にする
    numeric_columns = dataset.select_dtypes(include=[float, int]).columns
    numeric_columns = [col for col in target_emotions if col in numeric_columns]

    if not numeric_columns:
        print(f"No valid emotion columns found for mood: {mood}.")
        return render_template(f'{mood}.html', recommendations=[], graphs={})

    # 感情スコアを平均値で集約
    aggregated_emotions = dataset[numeric_columns].mean().to_dict()

    # 類似度計算
    similarities = []
    for _, row in dataset.iterrows():
        survey_vector = [aggregated_emotions.get(emotion, 0) for emotion in numeric_columns]
        dataset_vector = row[numeric_columns].values.astype(float)

        # ベクトルのノルムが0の場合はスキップ
        if np.linalg.norm(survey_vector) == 0 or np.linalg.norm(dataset_vector) == 0:
            similarities.append(0)
            continue

        # コサイン類似度を計算
        similarity = 1 - cosine(survey_vector, dataset_vector)
        similarities.append(similarity)

    # データセットに類似度を追加
    dataset['similarity'] = similarities
    print(f"Dataset with similarity scores:\n{dataset[['track_name', 'similarity']].head()}")

    # 上位5曲を取得
    top_matches = dataset.nlargest(100, 'similarity')

    past_recommendations = session.get('past_recommendations', [])
    new_recommendations = [
        match for _, match in top_matches.iterrows()
        if match['track_name'] not in past_recommendations
    ][:5]  # 5件までフィルタリング

    if new_recommendations:
        session['past_recommendations'] = past_recommendations + [
            match['track_name'] for match in new_recommendations
        ]

    # 推薦曲リストを生成
    def format_song(match):
        return f"{match['track_name']}（作曲者: {match['artists']}）"

    recommended_songs = [
        {
            "name": match['track_name'],
            "formatted": format_song(match),
            "scores": {emotion: match[emotion] for emotion in numeric_columns}
        }
        for match in new_recommendations
    ]

    # テンプレートに渡す
    return render_template(
        f'{mood}.html',
        recommendations=recommended_songs,
        emotion_colors=emotion_colors
    )


@app.template_filter('convert_to_float')
def convert_to_float(value):
    if isinstance(value, np.float32):
        return float(value)
    elif isinstance(value, dict):
        return {key: convert_to_float(val) for key, val in value.items()}
    elif isinstance(value, list):
        return [convert_to_float(item) for item in value]
    else:
        return value


if __name__ == '__main__':
    app.run(debug=True)


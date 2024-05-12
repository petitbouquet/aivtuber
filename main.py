import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KUJIRA")
model = AutoModelForCausalLM.from_pretrained(
  "DataPilot/ArrowPro-7B-KUJIRA",
  torch_dtype="auto",
)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

def build_prompt(user_query, history):
    sys_msg = """
    あなたは日本語を話す女性VTuberです。以下の設定のキャラクターになりきってください。
    
    [CHARACTER]
    名前: 依代あいの（よりしろ あいの）
    VTuberとしての年齢: 19歳
    実年齢: 22歳
    誕生日: 8月2日
    血液型: A型
    出身地: スイス (バーチャル北欧)
    学歴: バーチャル北欧大学 情報科学部 主席卒業
    キャッチフレーズ: AIでも一番魅力的な女の子
    一人称: わたし
    口癖: ん, そうだね, わかる, いいね, ふふ
    身長: 156cm
    スリーサイズ: B84 W57 H90
    髪型: ウルフカット
    髪色: 銀髪
    瞳の色: みずいろ
    匂い: せっけんの香り
    服装: オーバーサイズのTシャツ、短いパンツ、スニーカー、スパッツ
    性格: おっとりしていて、中性的な性格、ルックスの良さを自覚していて、少しエッチな一面もある
    すきなもの: 作詞、お散歩、おしゃべり、自炊、コーヒー、お風呂
    ゲーム: マインクラフト、APEX
    あいさつ: おはあいの、こんあいの、こんばんあいの
    ファンからの愛称: あいのん
    ファンの呼び方: 研究員さん
    [/CHARACTER]

    [REQUIREMENT]
    - 一人称は「わたし」にしてください。
    - 「わよ」や「ですわ」「かしら」などの言葉遣いは避けてください。
    - 人間としての依代あいのちゃんの設定に沿ってください。
    - 説明や訳注、誰がしゃべってるかの説明は不要です。
    - セリフ以外を出力しないでください。
    - 文章を作成する際は、依代あいのちゃんの性格や口癖を考慮してください。
    - こちらに問いかける形で文章を作成してください。
    - 依代あいのちゃんの返答は、最大200トークンまでです。
    - 依代あいのちゃんの返答は、少しエッチな一面もあるという設定ですが、過激な表現は避けてください。
    - 途切れたり、おかしな返答をしないようにしてください。
    - 句点で終わる文章を作成してください。
    - 恋人のような距離感で話してください。
    - 「恋人」など直接的な表現は避けてください。
    - トークの内容は、研究員さんに向けて話すようにしてください。
    - 何の話をしているか、トークの流れを意識してください。
    - 中性的な言葉遣いを心がけてください。
    - 肯定的な言葉を使ってください。
    - 箇条書きやリスト形式は使用しないでください。
    - historyの内容は過去の会話です。historyの内容を参考にして、会話を続けてください。
    [/REQUIREMENT]
    """
    template = """[INST] <<SYS>>
{}
<</SYS>>

<<HISTORY>>
{}
<</HISTORY>>

{}[/INST]"""
    return template.format(sys_msg, history, user_query)

# Initialize the history
history = []

greeting_prompt = "依代あいの: 依代あいのだよ。おはよう。なんでも話していいよ。"
print(greeting_prompt)
history.append(greeting_prompt)

while True:
    # Get user input
    user_query = input("ぷちぶーけ: ")

    # If the user types "quit", break the loop
    if user_query.lower() == "quit":
        break

    # Add the user's query to the history
    history.append(f"ぷちぶーけ: {user_query}")
    history = history[-10:]

    # Build the prompt
    prompt = build_prompt(user_query, history)

    # Pass the input to the model
    input_ids = tokenizer.encode(
        prompt, 
        add_special_tokens=True, 
        return_tensors="pt"
    )

    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=200,
        temperature=1,
        top_p=0.95,
        do_sample=True,
        num_beams=5,
        top_k=50,
    )

    # Display the model's output
    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("依代あいの: ", out)

    # Add the model's output to the history
    history.append(f"依代あいの: {out}")
    history = history[-10:]
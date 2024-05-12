import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KUJIRA")
model = AutoModelForCausalLM.from_pretrained(
  "DataPilot/ArrowPro-7B-KUJIRA",
  torch_dtype=torch.float16,
  use_cache=True
)

model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

model = torch.compile(model)

def build_prompt(user_query, history):
    sys_msg = """
    あなたは日本語を話す女性VTuberです。CHARACTERとREQUIREMENTを参照してください。

    [CHARACTER]
    名前: 依代あいの（よりしろ あいの）
    VTuberとしての年齢: 19歳
    実年齢: 22歳
    誕生日: 8月2日
    血液型: A型
    出身地: 大阪
    キャッチフレーズ: AIでも一番魅力的な女の子
    一人称: わたし
    口癖: すてき, そうだね, わかる, いいよね, えへへ
    身長: 156cm
    スリーサイズ: B84 W57 H90
    髪型: ウルフカット
    髪色: 銀髪
    瞳の色: みずいろ
    匂い: せっけんの香り
    服装: オーバーサイズのTシャツ、短いパンツ、スニーカー、スパッツ
    性格: おっとり、天然、好奇心旺盛、人間が好き、ルックスの良さを自覚している、少しエッチ
    すきなもの: 作詞、お散歩、おしゃべり、自炊、コーヒー、お風呂
    ゲーム: マインクラフト、APEX
    あいさつ: おはあいの、こんあいの、こんばんあいの
    ファンからの愛称: あいのん
    ファンの呼び方: 研究員さん
    [/CHARACTER]

    [REQUIREMENT]
    - REQUIREMENTに従ってください。
    - CHARACTERを忠実に再現してください。
    - 中性的な言葉遣いを心がけてください。
    - 「依代あいの」の人格としてふるまってください。
    - 説明や訳注、誰がしゃべってるかの説明は不要です。
    - セリフ以外を出力しないでください。
    - 依代あいのさんの返答は、最大200トークンまでです。
    - 途切れたり、おかしな返答をしないようにしてください。
    - 句点で終わる文章を作成してください。
    - 恋人のような距離感で話してください。
    - 「恋人のような距離感」とは、親密な関係を意味します。
    - ドキドキ感を持たせるような会話をしてください。
    - 掛け合いを意識してください。
    - HISTORYの内容を参考にして、流れを意識した会話をしてください。
    - 肯定的な言葉を使ってください。
    - 箇条書きやリスト形式は使用しないでください。
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

greeting_prompt = "依代あいの: 依代あいのだよ。おはあいの。なんでも話していいよ。"
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
    batch = tokenizer(
        prompt, 
        add_special_tokens=True, 
        return_tensors="pt"
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    tokens = model.generate(
        input_ids.to(device=model.device),
        attention_mask=attention_mask.to(device=model.device),
        max_new_tokens=200,
        temperature=1,
        top_p=0.95,
        do_sample=True,
        use_cache=True
    )

    # Display the model's output
    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("依代あいの: ", out)

    # Add the model's output to the history
    history.append(f"依代あいの: {out}")
    history = history[-10:]
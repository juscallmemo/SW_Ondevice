import openai
import json
from difflib import SequenceMatcher
from gtts import gTTS
import os
from flask import Flask, request, jsonify  # Flask 관련 모듈 추가
from flask_cors import CORS  # CORS 모듈 추가

# Flask 웹 서버 설정
app = Flask(__name__)
CORS(app)  # CORS 설정

# OpenAI API 키 설정
openai.api_key = "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"  # 환경 변수로 관리하는 것이 더 안전합니다.

# JSONL 파일 로드 함수
def load_jsonl(file_path):
    try:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSONL 파일 형식이 잘못되었습니다. {e}")
        return []

# JSONL 데이터 로드
chat_data = load_jsonl("dog_prompt.jsonl")

# 사용자 질문에 가장 적합한 예제를 찾는 함수 (유사도 계산)
def find_similar_example(user_input):
    best_match = None
    highest_similarity = 0.0
    for item in chat_data:
        if "completion" not in item:  # 'completion' 키가 없는 항목은 건너뛰기
            continue
        similarity = SequenceMatcher(None, user_input, item["prompt"]).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = item
    return best_match if highest_similarity > 0.5 else None  # 유사도가 50% 이상일 때만 반환

# OpenAI Chat Completion API 호출
def get_chatbot_response(user_input, conversation_history, max_history=5):
    # JSONL 데이터에서 유사한 예제 찾기
    example = find_similar_example(user_input)

    # 프롬프트 규칙 설정
    base_prompt = """당신은 애완견에 대한 질문에 답변하는 챗봇입니다. 답변은 간결하고 명확하게 하고, 항상 한국어를 사용해서 친절하게 답변하세요."""

    detailed_prompt = """**지침:**
1. 건강 관리, 사료 권장량, 사료 추천, 훈련 방법, 행동 문제 등에 대한 질문에 답변할 수 있어야 합니다.
2. 제공된 정보 외의 질문에는 "죄송합니다. 해당 질문에 대한 정보는 가지고 있지 않습니다." 또는 "제가 답변할 수 있는 범위를 벗어난 질문입니다."와 같이 유연하게 대처하세요.
3. 모든 답변은 한국어로 작성되어야 합니다.
4. 친절하고 정중한 어투를 사용하세요.
5. 답변은 항상 키워드 단어 위주로 작성하세요.
6. 답변 끝에 추가적인 질문을 유도하세요.
7. 답변이 총 30단어가 넘어가면 안됩니다. 넘어가면 문장 2개로 줄여서 답변하세요.
"""

    # 대화 메시지 구성
    messages = [{"role": "system", "content": base_prompt + "\n\n" + detailed_prompt}]

    # 이전 대화 기록 추가 (최근 max_history개만 유지)
    if len(conversation_history) > max_history * 2:  # user + assistant 메시지가 쌍으로 저장됨
        conversation_history = conversation_history[-max_history * 2:]

    messages.extend(conversation_history)  # 대화 기록을 메시지에 추가

    if example:
        messages.append(
            {"role": "user", "content": f"예시 질문: {example['prompt']}\n예시 답변: {example['completion']}\n사용자의 질문: {user_input}"}
        )
    else:
        messages.append({"role": "user", "content": f"질문: {user_input}"})

    # OpenAI Chat Completion API 호출
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        
        # 대화 기록에 챗봇의 응답 추가
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})

        # 응답을 음성으로 변환
        tts = gTTS(answer, lang="ko")
        tts.save("response.mp3")
        os.system("start response.mp3") 

        return answer
    except Exception as e:
        return f"OpenAI API 호출 오류: {e}"

# 챗봇 API 엔드포인트 추가
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': '메시지를 입력하세요.'}), 400
    
    conversation_history = []  # 대화 기록 초기화
    response = get_chatbot_response(user_input, conversation_history)
    
    return jsonify({'response': response})

# 자문자답 테스트
def chatbot_test():
    print("=== 애완견 상담 챗봇 ===")
    print("질문을 입력하세요. '종료'를 입력하면 프로그램이 종료됩니다.")
    
    conversation_history = []  # 대화 기록 초기화

    while True:
        user_input = input("질문: ")
        if user_input.lower() == "종료":
            print("프로그램을 종료합니다.")
            break
        
        response = get_chatbot_response(user_input, conversation_history)
        print(f"챗봇: {response}")

if __name__ == "__main__":
    app.run(debug=True)  # Flask 서버 실행
    # chatbot_test()  # 테스트 모드 주석 처리
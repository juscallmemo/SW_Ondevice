from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import sys
import site
import os
import random
from konlpy.tag import Okt
import time

# 시스템 정보 출력
print("Python 경로:", sys.executable)
print("Python 버전:", sys.version)
print("사이트 패키지 경로:", site.getsitepackages())

def download_and_save_model():
    print("모델 다운로드 시작...")
    model_name = "skt/kogpt2-base-v2"
    
    save_directory = "./saved_model"
    os.makedirs(save_directory, exist_ok=True)
    
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(save_directory)
        model = GPT2LMHeadModel.from_pretrained(save_directory)
        print("저장된 모델을 로드했습니다.")
    except:
        print("새로운 모델을 다운로드합니다...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        print("모델 저장 중...")
        tokenizer.save_pretrained(save_directory)
        model.save_pretrained(save_directory)
        print("모델 저장 완료")
    
    return tokenizer, model

class LightweightChatAgent:
    def __init__(self, model_path: str = "./saved_model"):
        print("모델 로딩 시작...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        print("토크나이저 로딩 완료")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        print("모델 로딩 완료")
        
    def export_to_onnx(self, path: str = "chat_agent.onnx"):
        print("ONNX 변환 시작...")
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                return self.model(input_ids)[0]

        wrapped_model = ModelWrapper(self.model)
        wrapped_model.eval()
        
        # GPT2는 더 긴 시퀀스를 처리할 수 있음
        dummy_input = torch.randint(1, 1000, (1, 256))
        
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            path,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            }
        )
        print("ONNX 변환 완료")

class OptimizedInference:
    def __init__(self, onnx_path: str):
        try:
            print("모델 초기화 시작...")
            # 메모리 최적화를 위한 설정
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.backends.cudnn.benchmark = True
            
            # KoGPT2 모델과 토크나이저 초기화
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                "./saved_model",
                model_max_length=128  # 최대 길이 제한
            )
            self.model = GPT2LMHeadModel.from_pretrained(
                "./saved_model"
            ).to(self.device)
            
            # 메모리 최적화를 위한 설정
            self.model.eval()  # 평가 모드로 설정
            torch.cuda.empty_cache()  # CUDA 캐시 정리
            
            # 패딩 토큰 명시적 설정
            special_tokens = {
                'pad_token': '[PAD]',
                'eos_token': '',
                'bos_token': '',
                'unk_token': ''
            }
            # 특수 토큰 추가
            
            self.tokenizer.add_special_tokens(special_tokens)
            # 모델 임베딩 크기 조정
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            print("KoGPT2 모델 로드 완료")
            
            # 데이터 파일의 마지막 수정 시간을 저장할 변수 추가
            self.data_file_path = "dog_data.txt"
            self.last_modified_time = None
            
            # 데이터 로드
            print("데이터 로드 시작...")
            self.knowledge_base = self.load_knowledge_base(self.data_file_path)
            
            # 문장 임베딩 계산 또는 로드
            print("문장 임베딩 처리 중...")
            self.compute_embeddings()
            print("초기화 완료")
            
            # konlpy 초기화 (형태소 분석기)
            self.okt = Okt()
            
            self.conversation_history = []
            self.context_window = 5  # 최근 5개 대화 유지
            
            self.question_analyzer = QuestionAnalyzer()
            
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            raise InitializationError("모델 초기화 실패")

    def load_knowledge_base(self, file_path: str) -> list:
        """데이터 파일에서 문장들을 로드"""
        sentences = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                for sent in text.split('.'):
                    sentences.extend(sent.split('!'))
                sentences = [s.strip() for s in sentences if s.strip()]
                print(f"{len(sentences)}개의 문장을 로드했습니다.")
                
        except FileNotFoundError:
            print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
            # 더 자연스러운 대화형 응답 추가
            sentences = [
                "안녕하세요! 강아지 훈련에는 칭찬과 보상이 매우 중요해요",
                "강아지와 함께 산책하면서 기본적인 명령어를 연습하는 게 좋아요",
                "강아지가 잘못된 행동을 할 때는 부정적인 반응보다 올바른 행동을 유도하는 게 효과적이에요",
                "매일 일정한 시간에 식사와 산책을 하면 강아지의 생활 패턴이 안정되어요",
                "강아지와 놀아주는 시간을 충분히 가지면 분리불안을 예방할 수 있어요"
            ]
            print("기본 대화 데이터를 사용합니다.")
        return sentences

    def get_embedding(self, text: str) -> np.ndarray:
        """KoGPT2를 사용하여 텍스의 임베딩을 얻음"""
        with torch.no_grad():
            # 입력 텍스트 토큰화
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding='max_length',
                max_length=128,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True
            )
            
            # 모델 통과
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                output_hidden_states=True
            )
            
            # 마지막 레이어의 hidden states 평균
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            masked_hidden = last_hidden_state * attention_mask
            sum_hidden = torch.sum(masked_hidden, dim=1)
            count_mask = torch.sum(attention_mask, dim=1)
            mean_hidden = sum_hidden / count_mask
            
            return mean_hidden[0].numpy()

    def compute_embeddings(self):
        """문장 임베딩을 계산하거나 저장된 것 로드"""
        embeddings_path = 'sentence_embeddings.npy'
        current_modified_time = os.path.getmtime(self.data_file_path)
        
        # 임베딩 파일이 존재하고, 데 파일이 변경되지 않았다면 기존 임베딩 사용
        if os.path.exists(embeddings_path):
            try:
                # 마지막 수정 시간 정보 로드
                with open('last_modified_time.txt', 'r') as f:
                    saved_modified_time = float(f.read().strip())
                
                # 데이터 파일이 변경되지 않았다면 저장된 임베딩 사용
                if saved_modified_time == current_modified_time:
                    print("저장된 임베딩 로드 중...")
                    self.embeddings = np.load(embeddings_path)
                    self.last_modified_time = saved_modified_time
                    print("저장된 임베딩 로드 완료")
                    return
            except:
                pass
        
        # 새로 계산하고 저장
        print("문장 임베딩 새로 계산 중...")
        self.embeddings = np.array([
            self.get_embedding(sent) 
            for sent in self.knowledge_base
        ])
        np.save(embeddings_path, self.embeddings)
        
        # 마지막 수정 시간 저장
        with open('last_modified_time.txt', 'w') as f:
            f.write(str(current_modified_time))
        self.last_modified_time = current_modified_time
        
        print("새로운 임베딩 저장 완료")

    def find_relevant_tokens(self, query: str) -> list:
        """쿼리와 관련된 토큰들을 찾아서 반환"""
        # 쿼리 토큰화
        query_tokens = self.tokenizer.tokenize(query)
        
        relevant_sentences = []
        for sentence in self.knowledge_base:
            # 문장 토큰화
            sent_tokens = self.tokenizer.tokenize(sentence)
            
            # 토 단위로 매칭 점수 계산
            match_score = 0
            matched_tokens = set()
            
            for q_token in query_tokens:
                for s_token in sent_tokens:
                    # 완전 일치하는 경우
                    if q_token == s_token:
                        match_score += 1.0
                        matched_tokens.add(s_token)
                    # 부분 일치하는 경우 (서브워드)
                    elif q_token in s_token or s_token in q_token:
                        match_score += 0.5
                        matched_tokens.add(s_token)
            
            if match_score > 0:
                relevant_sentences.append({
                    'sentence': sentence,
                    'score': match_score,
                    'matched_tokens': matched_tokens
                })
        
        return sorted(relevant_sentences, key=lambda x: x['score'], reverse=True)

    def extract_keywords(self, text: str) -> list:
        """텍스트에서 주요 키워드 추출"""
        # 명사, 동사, 형용사 추출
        words = self.okt.pos(text)
        keywords = []
        for word, pos in words:
            if pos in ['Noun', 'Verb', 'Adjective'] and len(word) > 1:
                keywords.append(word)
        return keywords
    
    def combine_sentences(self, keywords: list, num_samples: int = 3, temperature: float = 0.8) -> list:
        """키워드를 포함하는 문장들을 조합하여 새로운 문장 생성"""
        try:
            print("\n=== 문장 생성 프로세스 시작 ===")
            
            # 1. 키워드 관련 문장 검색
            print("\n1. 키워드 관련 문장 검색...")
            relevant_sentences = []
            for keyword in keywords:
                print(f"→ '{keyword}' 키워드 검색 중...")
                for sentence in self.knowledge_base:
                    if keyword in sentence:
                        relevant_sentences.append(sentence)
                        print(f"  ✓ 발견: {sentence}")
            
            print(f"\n→ 총 {len(relevant_sentences)}개의 관련 문장 발견")
            
            if not relevant_sentences:
                print("→ 관련 문장을 찾을 수 없습니다.")
                return ["워드와 관련된 문장을 찾을 수 없습니다."]
            
            # 2. 문장 분리 및 조각 생성
            print("\n2. 문장 조각 생성...")
            sentence_parts = []
            for sent in relevant_sentences:
                parts = sent.split(',')
                cleaned_parts = [p.strip() for p in parts if p.strip()]
                sentence_parts.extend(cleaned_parts)
                print(f"→ 분리된 조각들: {cleaned_parts}")
            
            print(f"\n→ 총 {len(sentence_parts)}개의 문장 조각 생성됨")
            
            # 3. 새로운 문장 조합
            print("\n3. 새로운 문장 조합 시작...")
            new_sentences = []
            for i in range(num_samples):
                if len(sentence_parts) < 2:
                    continue
                
                print(f"\n→ {i+1}번째 문장 생성:")
                
                # 문장 조각 선택
                num_parts = random.randint(2, min(3, len(sentence_parts)))
                print(f"  ▷ {num_parts}개의 조각 선택 (temperature: {temperature:.2f})")
                
                selected_parts = []
                for j in range(num_parts):
                    part = self.weighted_choice(sentence_parts, temperature)
                    selected_parts.append(part)
                    print(f"  ▷ 선택된 조각 {j+1}: {part}")
                
                # 접속사 선택 및 문장 조합
                connectors = ['그리고 ', '또한 ', '또 ', '그래서 ', '', '게다가 ', '특히 ']
                
                new_sentence = selected_parts[0]
                for part in selected_parts[1:]:
                    connector = self.weighted_choice(connectors, temperature)
                    print(f"  ▷ 선택된 접속사: '{connector}'")
                    new_sentence += f", {connector}{part}"
                
                print(f"  ▷ 생성된 문장: {new_sentence}")
                new_sentences.append(new_sentence)
            
            print("\n=== 문장 생성 완료 ===")
            return new_sentences
            
        except Exception as e:
            print(f"문장 생성 중 오류 발생: {e}")
            raise SentenceGenerationError("문장 생성 실패")

    def calculate_similarity(self, query: str) -> list:
        """쿼리와 문장들 간의 코사인 유사도 계산"""
        query_embedding = self.get_embedding(query)
        
        # 코사인 유사도 계산
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 유사도가 높은 순서대로 정렬된 인덱스
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 상위 유사 문장들 반환
        return [
            {
                'sentence': self.knowledge_base[idx],
                'similarity': similarities[idx]
            }
            for idx in sorted_indices[:3]  # 상위 3개만 반환
        ]

    def generate_response(self, prompt: str) -> str:
        try:
            print("\n=== 응답 생성 프로세스 시작 ===")
            
            # 1. 질문 분석
            print("\n1. 질문 분석 중...")
            question_info = self.question_analyzer.analyze_intent(prompt)
            print(f"→ 질문 유형: {question_info['type']}")
            print(f"→ 핵심 키워드: {', '.join(question_info['keywords'])}")
            print(f"→ 질문 깊이: {question_info['depth']}")
            
            # 2. 답변 후��� 생성
            print("\n2. 답변 후보 생성 중...")
            candidates = self.combine_sentences(
                question_info['keywords'],
                num_samples=3,
                temperature=0.8
            )
            
            # 3. 최적 답변 선택
            print("\n3. 최적 답변 선택 중...")
            best_response = self.select_best_response(prompt, candidates)
            
            # 4. 답변 품질 향상
            print("\n4. 답변 품질 향상 중...")
            final_response = self.enhance_response_quality(prompt, best_response)
            
            print("\n=== 응답 생성 완료 ===")
            return final_response
            
        except Exception as e:
            print(f"응답 생성 중 오류 발생: {e}")
            return "죄송합니다. 답변 생성 중 문제가 발생했습니다."

    def select_best_response(self, question: str, candidates: list) -> str:
        """최적의 답변 선택"""
        try:
            # 각 후보 답변의 점수 계산
            scored_responses = []
            for response in candidates:
                score = self.calculate_relevance_score(question, response)
                scored_responses.append({
                    'response': response,
                    'score': score
                })
            
            # 점수순 정렬
            scored_responses.sort(key=lambda x: x['score'], reverse=True)
            print(f"→ 최고 점수 답변 선택됨 (점수: {scored_responses[0]['score']:.2f})")
            
            return scored_responses[0]['response']
            
        except Exception as e:
            print(f"답변 선택 중 오류: {e}")
            return candidates[0] if candidates else "답변을 생성할 수 없습니다."

    def calculate_relevance_score(self, question: str, answer: str) -> float:
        """답변의 관련성 점수 계산"""
        try:
            # 1. 질문 분석
            q_intent = self.question_analyzer.analyze_intent(question)
            
            # 2. 임베딩 기반 유사도 계산 (0.3)
            embedding_score = self.calculate_embedding_similarity(
                question, 
                answer
            )
            
            # 3. 키워드 매칭 점수 (0.4)
            keyword_score = self.calculate_keyword_matching(
                q_intent['keywords'], 
                answer
            )
            
            # 4. 문맥 적절성 점수 (0.3)
            context_score = self.evaluate_context_relevance(
                question, 
                answer
            )
            
            # 가중치 적용
            final_score = (
                embedding_score * 0.3 +
                keyword_score * 0.4 +
                context_score * 0.3
            )
            
            print(f"→ 점수 상세:")
            print(f"  - 임베딩 유사도: {embedding_score:.2f}")
            print(f"  - 키워드 매칭: {keyword_score:.2f}")
            print(f"  - 문맥 적절성: {context_score:.2f}")
            print(f"  - 최종 점수: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            print(f"관련성 점수 계산 중 오류: {e}")
            return 0.0

    def evaluate_context_relevance(self, question: str, answer: str, conversation_history: list = None) -> float:
        """문맥 적절성 평가"""
        try:
            # 1. 질문 유형에 따른 기대 패턴
            q_type = self.question_analyzer.identify_question_type(question)
            
            # 2. 패턴 매칭 점수 (0.6)
            patterns = {
                '방법': ['방법', '하세요', '합니다', '하면'],
                '이유': ['때문', '이유', '에요', '입니다'],
                '설명': ['입니다', '에요', '것은'],
                '시기': ['때', '시기', '경우'],
                '상태': ['상태', '증상', '모습']
            }
            
            pattern_score = 0.0
            if q_type in patterns:
                matches = sum(1 for p in patterns[q_type] if p in answer)
                pattern_score = min(matches / len(patterns[q_type]), 1.0)
            
            # 3. 대화 맥락 점수 (0.4)
            context_score = 0.7  # 기본 점수
            if conversation_history:
                try:
                    # 최근 대화와의 연관성 확인
                    recent_context = conversation_history[-1]
                    context_keywords = self.question_analyzer.extract_core_keywords(recent_context)
                    context_matches = sum(1 for k in context_keywords if k in answer)
                    context_score = min(context_matches / max(len(context_keywords), 1), 1.0)
                except:
                    pass
            
            # 4. 최종 점수 계산
            final_score = pattern_score * 0.6 + context_score * 0.4
            
            print(f"  ▷ 패턴 점수: {pattern_score:.2f}")
            print(f"  ▷ 맥락 점수: {context_score:.2f}")
            print(f"  ▷ 최종 문맥 점수: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            print(f"문맥 평가 중 오류: {e}")
            return 0.5

    def enhance_response_quality(self, question: str, response: str) -> str:
        """답변 품질 향상"""
        try:
            # 1. 질문 의도에 맞게 답변 구조화
            q_intent = self.question_analyzer.analyze_intent(question)
            
            if q_intent['type'] == '방법':
                response = self.structure_how_to_answer(response)
            elif q_intent['type'] == '이유':
                response = self.structure_explanation(response)
            
            # 2. 구체적인 예시 추가
            if '예시' not in response.lower():
                examples = self.find_relevant_examples(q_intent['keywords'])
                if examples:
                    response += f"\n\n예를 들면, {examples[0]}"
            
            # 3. 실행 가능한 조언 추가
            if q_intent['type'] in ['방법', '해결']:
                actionable_tips = self.generate_actionable_tips(response)
                response += f"\n\n실천 팁: {actionable_tips}"
            
            return response
            
        except Exception as e:
            print(f"답변 품질 향상 중 오류: {e}")
            return response

    def update_conversation_history(self, prompt: str):
        """대화 기록 업데이트 및 관리"""
        try:
            if not hasattr(self, 'conversation_history'):
                self.conversation_history = []
            
            # 최근 5개의 대화만 유지
            self.conversation_history = (self.conversation_history + [prompt])[-5:]
            
        except Exception as e:
            print(f"대화 기록 업데이트 중 오류 발생: {e}")

    def weighted_choice(self, items: list, temperature: float = 0.8) -> str:
        """temperature를 사용하여 가중치 기반 선택을 수행"""
        try:
            if not items:
                raise ValueError("선택할 항목이 없습니다.")
                
            # 가중치 계산 (temperature가 높을수록 더 균일한 분포)
            weights = [1.0 / (i + 1) ** (1.0 / temperature) for i in range(len(items))]
            
            # 가중치 정규화
            total = sum(weights)
            weights = [w / total for w in weights]
            
            # 가중치 반 선택
            selected = random.choices(items, weights=weights, k=1)[0]
            
            return selected
            
        except Exception as e:
            print(f"가중치 선택 중 오류 발생: {e}")
            # 오류 발생 시 랜덤 선택으로 폴백
            return random.choice(items)

    def manage_context(self, prompt: str, response: str):
        """대화 컨텍스트 관리"""
        self.conversation_history.append({
            'prompt': prompt,
            'response': response,
            'timestamp': time.time()
        })
        # 최근 대화만 유지
        self.conversation_history = self.conversation_history[-self.context_window:]

    def create_enhanced_prompt(self, user_input: str) -> str:
        """컨텍스트를 고려한 프롬프트 생성"""
        context = "\n".join([
            f"User: {conv['prompt']}\nAssistant: {conv['response']}"
            for conv in self.conversation_history[-3:]  # 최근 3개 대화만 포함
        ])
        
        return f"""
이전 대화:
{context}

현재 상황: 강아지 관련 전문가로서 친절하게 답변합니다.
사용자 질문: {user_input}
답변:"""

    def enhance_response(self, response: str) -> str:
        """응답 품질 개선"""
        # 1. 문장 자연스럽게 만들기
        response = self.improve_fluency(response)
        
        # 2. 맥락 일관성 확인
        if not self.check_consistency(response):
            response = self.regenerate_response()
        
        # 3. 적절한 감정/톤 추가
        response = self.add_emotional_tone(response)
        
        return response

    def improve_fluency(self, text: str) -> str:
        """문장 자연스럽게 만들기"""
        # 문장 끝맺음 개선
        endings = ['요', '니다']
        if not any(text.endswith(end) for end in endings):
            text += '요'
        
        # 접속사 자연스럽게
        text = text.replace(', 그리고', '고')
        text = text.replace(', 또한', '. 또한')
        
        return text

    def manage_conversation_flow(self, prompt: str) -> str:
        """대화 흐름 관리"""
        # 1. 이전 대화 참조
        related_history = self.find_related_conversations(prompt)
        
        # 2. 대화 단계 파악
        conversation_state = self.determine_conversation_state()
        
        # 3. 적절한 응답 전략 선택
        if conversation_state == 'initial':
            return self.generate_initial_response(prompt)
        elif conversation_state == 'followup':
            return self.generate_followup_response(prompt, related_history)
        else:
            return self.generate_closing_response(prompt)

    def expand_knowledge(self, base_response: str) -> str:
        """응답 내용 풍부화"""
        try:
            # 1. 핵심 개념 추출
            key_concepts = self.extract_key_concepts(base_response)
            
            # 2. 관련 지식 검색
            additional_info = self.search_related_knowledge(key_concepts)
            
            # 3. 응답 보강
            enhanced_response = f"{base_response}\n\n추가로, {additional_info}"
            
            return enhanced_response
            
        except Exception as e:
            print(f"지식 확장 중 오류 발생: {e}")
            return base_response

    def add_empathy(self, prompt: str, response: str) -> str:
        """공감 요소 추가"""
        # 감정 분석
        emotion = self.analyze_emotion(prompt)
        
        # 감정에 따른 공감 표현 추가
        empathy_phrases = {
            'concern': '걱정이 되시는군요. ',
            'curiosity': '궁금하신 점이 많으시군요. ',
            'frustration': '어려움을 겪고 계시는군요. ',
            'excitement': '기대가 크시군요. '
        }
        
        prefix = empathy_phrases.get(emotion, '')
        return f"{prefix}{response}"

    def calculate_relevance_score(self, question: str, candidate_answer: str) -> float:
        """답변의 관련성 점수 계산"""
        try:
            # 1. 질문 분석
            q_intent = self.question_analyzer.analyze_intent(question)
            
            # 2. 임베딩 기반 유사도 계산
            embedding_score = self.calculate_embedding_similarity(
                question, 
                candidate_answer
            )
            
            # 3. 키워드 매칭 점수
            keyword_score = self.calculate_keyword_matching(
                q_intent['keywords'], 
                candidate_answer
            )
            
            # 4. 문맥 적절성 점수
            context_score = self.evaluate_context_relevance(
                question, 
                candidate_answer, 
                self.conversation_history
            )
            
            # 가중치 적용
            final_score = (
                embedding_score * 0.4 +
                keyword_score * 0.3 +
                context_score * 0.3
            )
            
            return final_score
            
        except Exception as e:
            print(f"관련성 점수 계산 중 오류: {e}")
            return 0.0

    def select_best_response(self, question: str, candidate_responses: list) -> str:
        """가장 적합한 답변 선택 및 개선"""
        try:
            # 1. 각 답변의 점수 계산
            scored_responses = [
                {
                    'response': resp,
                    'score': self.calculate_relevance_score(question, resp)
                }
                for resp in candidate_responses
            ]
            
            # 2. 점수순 정렬
            scored_responses.sort(key=lambda x: x['score'], reverse=True)
            
            # 3. 최고 점수 답변 선택
            best_response = scored_responses[0]['response']
            
            # 4. 답변 개선
            enhanced_response = self.enhance_response_quality(
                question,
                best_response
            )
            
            return enhanced_response
            
        except Exception as e:
            print(f"답변 선택 중 오류: {e}")
            return candidate_responses[0]  # 폴백: 첫 번째 답변 반환

    def enhance_response_quality(self, question: str, response: str) -> str:
        """답변 품질 향상"""
        try:
            # 1. 질문 의도에 맞게 답변 구조화
            q_intent = self.question_analyzer.analyze_intent(question)
            
            if q_intent['type'] == '방법':
                response = self.structure_how_to_answer(response)
            elif q_intent['type'] == '이유':
                response = self.structure_explanation(response)
            
            # 2. 구체적인 예시 추가
            if '예시' not in response.lower():
                examples = self.find_relevant_examples(q_intent['keywords'])
                if examples:
                    response += f"\n\n예를 들면, {examples[0]}"
            
            # 3. 실행 가능한 조언 추가
            if q_intent['type'] in ['방법', '해결']:
                actionable_tips = self.generate_actionable_tips(response)
                response += f"\n\n실천 팁: {actionable_tips}"
            
            return response
            
        except Exception as e:
            print(f"답변 품질 향상 중 오류: {e}")
            return response

    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간의 임베딩 유사도 계산"""
        try:
            # 간단한 자카드 유사도 계산으로 대체
            set1 = set(text1.split())
            set2 = set(text2.split())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            if union == 0:
                return 0.0
                
            return intersection / union
            
        except Exception as e:
            print(f"임베딩 유사도 계산 중 오류: {e}")
            return 0.0

    def structure_explanation(self, response: str) -> str:
        """설명형 답변 구조화"""
        try:
            # 1. 문장 분리
            sentences = response.split('. ')
            
            # 2. 구조화
            if len(sentences) > 1:
                main_point = sentences[0]
                details = '. '.join(sentences[1:])
                
                structured = f"{main_point}. \n구체적으로, {details}"
            else:
                structured = response
            
            # 3. 마무리 추가
            structured += "\n이해가 되셨나요?"
            
            return structured
            
        except Exception as e:
            print(f"설명 구조화 중 오류: {e}")
            return response

    def structure_how_to_answer(self, response: str) -> str:
        """방법 설명형 답변 구조화"""
        try:
            # 1. 문장 분리
            sentences = response.split('. ')
            
            # 2. 단계별 구조화
            if len(sentences) > 1:
                intro = sentences[0]
                steps = sentences[1:]
                
                structured = f"{intro}.\n\n단계별 방법:\n"
                for i, step in enumerate(steps, 1):
                    structured += f"{i}. {step.strip()}.\n"
            else:
                structured = response
            
            return structured
            
        except Exception as e:
            print(f"방법 구조화 중 오류: {e}")
            return response

    def find_relevant_examples(self, keywords: list) -> list:
        """키워드와 관련된 예시 찾기"""
        try:
            examples = []
            for keyword in keywords:
                # knowledge_base에서 '예시' 또는 '예를 들어' 포함된 문장 찾기
                for sentence in self.knowledge_base:
                    if keyword in sentence and ('예시' in sentence or '예를 들어' in sentence):
                        examples.append(sentence)
            
            return examples[:2]  # 최대 2개의 예시 반환
            
        except Exception as e:
            print(f"예시 검색 중 오류: {e}")
            return []

    def generate_actionable_tips(self, response: str) -> str:
        """실행 가능한 팁 생성"""
        try:
            tips = [
                "매일 일정한 시간에 연습하세요",
                "긍정적인 강화를 사용하세요",
                "인내심을 가지고 천천히 진행하세요",
                "전문가와 상담하는 것도 좋은 방법입니다"
            ]
            
            # 응답 내용에 따라 관련된 팁 선택
            selected_tips = []
            for tip in tips:
                if any(keyword in response for keyword in ['훈련', '연습', '학습']):
                    if '시간' in tip or '연습' in tip:
                        selected_tips.append(tip)
                elif any(keyword in response for keyword in ['문제', '해결', '개선']):
                    if '전문가' in tip or '방법' in tip:
                        selected_tips.append(tip)
            
            return ' '.join(selected_tips) if selected_tips else tips[0]
            
        except Exception as e:
            print(f"팁 생성 중 오류: {e}")
            return "천천히 차근차근 진행해보세요."

    def calculate_keyword_matching(self, keywords: list, text: str) -> float:
        """키워드 매칭 점수 계산"""
        try:
            if not keywords or not text:
                return 0.0
            
            # 1. 키워드 전처리
            keywords = [k.lower() for k in keywords]
            text = text.lower()
            
            # 2. 매칭 점수 계산
            matches = 0
            total_weight = len(keywords)
            
            for i, keyword in enumerate(keywords):
                # 키워드 가중치 (첫 번째 키워드가 더 중요)
                weight = 1.0 / (i + 1)
                
                # 완전 일치
                if keyword in text:
                    matches += weight
                    continue
                
                # 부분 일치 (키워드가 2글자 이상인 경우)
                if len(keyword) >= 2:
                    for part in self.generate_substrings(keyword):
                        if part in text:
                            matches += weight * 0.5
                            break
            
            # 3. 정규화된 점수 반환 (0~1 사이)
            return min(matches / total_weight, 1.0)
            
        except Exception as e:
            print(f"키워드 매칭 점수 계산 중 오류: {e}")
            return 0.0

    def generate_substrings(self, text: str, min_length: int = 2) -> list:
        """텍스트의 부분 문자열 생성"""
        substrings = []
        text_length = len(text)
        
        for length in range(min_length, text_length + 1):
            for start in range(text_length - length + 1):
                substring = text[start:start + length]
                substrings.append(substring)
        
        return substrings

# 커스텀 예외 클래스들
class InitializationError(Exception):
    pass

class KeywordExtractionError(Exception):
    pass

class SentenceGenerationError(Exception):
    pass

class ResponseGenerationError(Exception):
    pass

class QuestionAnalyzer:
    def __init__(self):
        self.question_patterns = {
            '방법': ['어떻게', '방법', '할까', '하려면'],
            '이유': ['왜', '이유', '때문에'],
            '설명': ['뭔가요', '무엇', '뭐', '알려주세요'],
            '시기': ['언제', '시기', '때'],
            '상태': ['상태', '증상', '어떤']
        }
        self.okt = Okt()

    def analyze_intent(self, question: str) -> dict:
        """질문의 의도와 유형을 분석"""
        try:
            # 1. 질문 유형 파악
            q_type = self.identify_question_type(question)
            
            # 2. 핵심 키워드 추출
            keywords = self.extract_core_keywords(question)
            
            # 3. 질문 깊이 분석
            depth = self.analyze_question_depth(question)
            
            return {
                'type': q_type,
                'keywords': keywords,
                'depth': depth
            }
            
        except Exception as e:
            print(f"질문 분석 중 오류: {e}")
            return {'type': '설명', 'keywords': [], 'depth': 'basic'}

    def identify_question_type(self, question: str) -> str:
        """질문의 유형을 식별"""
        for q_type, patterns in self.question_patterns.items():
            if any(pattern in question for pattern in patterns):
                return q_type
        return '설명'  # 기본 유형

    def extract_core_keywords(self, question: str) -> list:
        """핵심 키워드 추출"""
        # 형태소 분석
        words = self.okt.pos(question)
        keywords = []
        for word, pos in words:
            # 명사, 동사, 형용사 추출
            if pos in ['Noun', 'Verb', 'Adjective'] and len(word) > 1:
                keywords.append(word)
        return keywords

    def analyze_question_depth(self, question: str) -> str:
        """질문의 깊이 분석"""
        # 질문의 길이와 복잡성을 기반으로 깊이 판단
        if len(question) > 30 or '왜' in question:
            return 'advanced'
        elif len(question) > 15:
            return 'intermediate'
        return 'basic'

if __name__ == "__main__":
    # 모델이 없을 경우에만 다운로드
    if not os.path.exists("./saved_model"):
        download_and_save_model()
    
    print("1. 모델 초기화 시작")
    agent = LightweightChatAgent()
    agent.export_to_onnx()

    print("\n2. 추론 엔진 초기화")
    inference_engine = OptimizedInference("chat_agent.onnx")

    print("\n3. 대화 시작")
    print("(dog_data.txt 파일의 정보를 기반으로 답변합니다)")
    
    while True:
        user_input = input("\n질문을 입력하세요 (종료하려면 'q' 입력): ")
        if user_input.lower() == 'q':
            break
            
        print("\n처리 중...")
        response = inference_engine.generate_response(user_input)
        print("답변:", response)

    print("\n대화를 종료합니다.")
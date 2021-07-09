import numpy as np
import hgtk

# reference : https://namu.wiki/w/한글/자모
jamo_list = ["ᴥ", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
jamo_list += ["ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"]
jamo_list += ["ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅄ"]
jamo_list += ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"]
jamo_list += ["ㅐ", "ㅒ", "ㅔ", "ㅖ", "ㅘ", "ㅙ", "ㅚ", "ㅝ", "ㅞ", "ㅟ", "ㅢ"]

n_symbols = 256 + len(jamo_list) # UTF-8 + jamo_list

def text_to_tokens(text):
    text_decomposed = hgtk.text.decompose(text)
    tokens = []
    for t in text_decomposed:
        if t in jamo_list:
            index = jamo_list.index(t)
            tokens.append(256 + index)
        else:
            encoded = t.encode()
            for e in encoded:
                tokens.append(e)
                
    return np.array(tokens)

def tokens_to_text(tokens):
    def token_to_char(token):
        try:
            char = np.array([token]).astype(np.int8).tobytes().decode()
            return char
        except:
            return ""
    
    text = ""
    for t in tokens:
        if t < 256:
            char = token_to_char(t)
        else:
            char = jamo_list[t - 256]
        text += char
    text = hgtk.text.compose(text)
    
    return text

def remove_double_script(text):
    '''
    1.1.1. 표준발성에서 벗어나거나 같은 전사에 대하여 두 가지 이상 발음이 가능한 경우 발
    음전사와 철자전사를 병행하며, 이 경우 (철자전사)/(발음전사)로 표기한다 (이 문서에
    서 향후 이를 '이중전사'라 칭한다).
    '''
    
    index = text.find(')/(')
    if index < 0:
        return text
    
    left_text = text[:index]
    right_text = text[index+3:]
    
    left_last_index = left_text.find('(')
    left_text = left_text[:left_last_index]
    
    right_last_index = right_text.find(')')
    right_text = right_text[:right_last_index] + right_text[right_last_index+1:]
    
    text = left_text + right_text
    text = remove_double_script(text)
    
    return text
    
def remove_details(text):
    '''
    1.2.3. 다음에 정의된 잡음 이름 뒤에 ‘/’를 붙여 표기한다.
    - b : 숨소리
    - l : 웃음 소리(laugh)
    - o : 다른 사람의 말소리가 포함된 경우 문장의 맨 앞에 표기
    - n : 주변의 잡음
    1.9.1. 화자가 발음한 내용을 잘 알아 듣기 힘들 때 어절의 뒷부분에 ‘*’를 붙여 이중전사한다.
    1.4.1. 발성자가 다음 발성을 준비하기 위해서 소요되는 시간을 벌기 위해서 발성하는 것으
          로 의미 없는 것을 말한다. 간투어 뒤에 ‘/’를 붙여 표기한다.
    1.9.3. 문맥을 고려해봐도 전혀 알아들을 수 없는 발화는 ‘u/’ 으로 표기한다.      
    1.9.4. 발성과 동시에 발생하는 잡음은 어절 끝에 ‘*’를 붙여 표기한다.
    1.9.5. 반복 발성이나 잘못된 발성은 반드시 표기 한다. 이때 불필요하게 중복 또는 잘못 발성된
           부분은 뒤에 ‘+’를 붙인다. 예) 아침에 학교+ 학교에 갔다
    '''
    details = ['b/', 'l/', 'o/', 'n/', '*', 'u/', '/', '+']
    for d in details:
        text = text.replace(d, '')
        
    return text

def remove_double_space(text):
    if text.find('  ') < 0:
        return text

    text = text.replace('  ', ' ')
    text = remove_double_space(text)    
    
    return text
    
def refine_ksponspeech(text):
    text = remove_double_script(text)
    text = remove_details(text)
    text = remove_double_space(text)
    text = text.strip()
    
    return text
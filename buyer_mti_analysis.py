import pyodbc 
import os
import glob
import pandas as pd
import numpy as np
import pickle
import re
from tqdm import tqdm
from ast import literal_eval
from itertools import chain
from collections import Counter, defaultdict
from konlpy.tag import Komoran
import nltk
import time
from datetime import datetime

# from config import *

tqdm.pandas()

class PreProcess:
	def __init__(self):
		self.ko_stopwords = ['여러가지', '직접', '경영', '사회', '수출입', '상품', '앞쪽', '업체', '은닉', '현황', '구입', '다툼', '예외', '전문', '효능', '전문가', '국내외', '취급', '토대', '애원', '범위', '자신감', '정보', '소요', '조사', '결정', '최고', '조언', '향상', '허용', '도달', '목록', '홈페이지', '대부분', '형태', '추진구축', '임시', '지정', '이메일', '목표', '유출', '법인', '퇴장', '포함', '조치', '희망', '업무', '도모', '방법', '투자', '극복', '방해', '각종', '제안', '위협', '혼잡', '전부', '판매', '해당', '손실', '인증', '완전', '실현', '다수', '지점', '제시', '초과', '검토', '목적', '고급', '브랜드', '요망', '각광', '상담주선', '제호', '참가', '부탁', '신경', '경우', '특징', '부분', '정책', '완화', '사업', '분야', '관련', '기대', '학년', '그룹', '평소', '참조', '존재', '단계', '그중', '회의', '올해', '직영', '대응', '즉각', '본사', '자영업자', '설립', '소개', '코드', '대책', '제작', '과장', '행복', '비교', '과장', '일자', '일반', '연결', '관리자', '제품', '설명서', '관심', '과거', '분류', '접근', '이번', '용도', '설정', '주최', '주선', '바람', '내용', '고객', '정부', '개최', '발신', '수단', '획득', '프로젝트', '기타로', '대체', '조금', '메일', '바이어', '첨부', '상담', '요청', '드림', '제조', '기타', '부품', '백색', '용품','광고', '한국', '도구', '현지', '기기', '기계', '소형','선정','교류', '전반', '기획', '개발', '생산',' 기초', '자동', '수동', '장비', '기기', '제어', '실내', '실외', '기능', '산업','협회', '품목', '라인', '가공', '냉장', '냉동', '기관', '시장', '가공품', '혁신', '시스템', '물자', '비품', '담당', '물류', '원료', '특수', '설계', '처리', '화학', '보제', '서비스', '보조', '관리', '운영', '최종', '테스트', '대형', '중형', '설치', '필요', '현대', '입력', '교육', '유통', '기업', '구매', '구입', '국내', '해외', '전자', '자동', '수동', '남성', '여성', '성인', '이벤트', '조건', '유지', '보수', '선정', '중고', '모듈', '파트', '어이', '대행', '순간', '색상', '청색', '녹색', '흑색', '완료', '확인', '키워드', '기반', '자료', '보완', '효과', '니다', '소재', '동시', '가능', '제외', '열거', '크기', '주변기기', '기능성', '고무']

		self.en_stopwords = ['making','working','instrument','instruments','tteopoki', 'module', 'veau', 'omputer', 'consu', 'composition', 'than', 'wwwmobilegameskr', 'apple', 'won', 'management', 'haven', 'enable', 'platform', 'themselves', 'process', 'fund', 'combi', 'ools', 'which', 'having', 'cpnp', "hadn't", 'contact', 'acces', 'processing', 'how', 'black', 'hindu', 'aaaaa', 'rulo', 'are', 'during', 'further', 'dcor', 'll', 'off', "you'll", 'korea', 'oinp', 'basic', 'unit', 'mightn', 'y', 'wasn', "needn't", 'hasn', 'at', 'personal', 'secur', 'whol', 'with', 'schoo', 'yourself', 'such', 'me', 'all', 'individual', 'application', 'what', 'reade', 'suppleme', 'type', 'khai', 't', 'test', 'both', 'machine', 'night', 'until', 'needn', 'dd', 'inch', 're', 'his', 'element', 'free', 'ours', 'itself', 'shop', 'why', 'cc', 'songwha', "mightn't", 'rare', 'but', 'few', 'in', 'be', 'himself', 'odor', 'x', 'between', 'through', 'very', 'does', 'public', 'push', 'detect', 'only', 'to', 'by', 'yumins', 'lase', 'we', 'solu', 'theirs', 'your', "aren't", 'was', 'couldn', 'hadn', 'substance', 'has', 'blow', 'color', 'ghee', 'toppokki', "wasn't", 'hipaa', 'part', 'being', 'produce', 'ther', 'nces', 'my', 'applemobilegameskr', 'supp', 'range', "should've", 'coco', "shouldn't", 'aren', 'cali', 'he', 'hour', 'them', 'ourselves', 'const', 'tool', 'article', 'can', 'engg', 'jeus', 'electro', 'against', 'none', 'semi', 'swcc', 'should', 'etcwwwwoorirocom', 'other', 'woorim', 'reed', 'devi', 'their', 'data', 'middle', 'as', 'schoool', 'mine', 'mfrs', 'mechanical', 'hers', 'company', 'nori', 'held', 'text', 'etc', 'posco', 'these', 'leisur', 'scre', 'user', 'inc', 'para', "doesn't", 'prop', 'stee', 's', 'alll', 'hansung', 'own', "wouldn't", 'mechanism', 'change', 'just', 'yang', 'yours', 'inje', 'a', 'input', 'ma', 'other', 'now', 'down', 'so', 'cisr', 've', "you'd", 'supe', 'not', 'art', 'track', 'block', 'shan', 'ain', 'who', 'out', 'cold', 'others', 'engin', 'model', 'she', 'stud', 'each', 'white', 'isn', 'where', 'feng', 'prod', 'access', 'producer', 'company', 'wwwakgreentechcokr', 'size', 'nsk', 'you', 'include', 'weren', 'there', 'machinery', 'have', 'more', 'poly', 'noon', 'elect', 'raw', 'otocs', 'above', "you've", 'sudarat', 'tracotr', 'some', 'esco', "mustn't", 'nor', 'ex', 'smart', 'under', 'brown', 'am', 'cosm', 'nec', 'most', 'click', 'others', 'anti', "that'll", 'japan', 'syst', 'below', 'farma', 'were', 'of', "couldn't", 'eriak', "won't", 'myself', 'regular', 'an', 'natural', 'sungl', 'will', 'from', 'poct', 'because', 'its', 'up', 'mgmt', 'too', 'that', "don't", 'check', 'artificial', 'stop', 'do', "isn't", 'i', 'produ', 'measurement', 'and', 'blank', 'kotra', 'mustn', 'ldpe', 'machining', 'no', 'receiver', 'dzeus', 'about', 'into', 'herself', 'production', 'cond', 'press', 'base', 'yourselves', 'universal', 'lg', 'x', 'it', 'the', 'over', "she's", 'grade', "weren't", "haven't", 'before', 'applia', 'oems', 'her', 'othe', 'hard', 'etc', 'whom', 'step', 'incen', 'd', 'basf', 'elec', "didn't", 'traditional', 'if', 'our', 'then', 'ap', 'when', 'to', 'chia', 'shouldn', 'tester', 'doesn', 'mport', 'same', 'm', 'coopera', 'good', 'consumabl', 'ltd', 'those', 'wwwisavecokr', 'any', 'anna', 'again', 'o', 'zeus', 'info', 'pmma', 'trans', 'here', 'live', 'once', "you're", 'blue', 'him', 'this', 'tran', 'for', 'after', 'did', 'eqip', 'product', 'special', 'case', 'don', "shan't", 'had', 'juan', 'powder', 'conte', 'industry', 'drin', 'valu', 'on', 'work', 'wouldn', 'is', 'doing', 'tele', 'installation', "hasn't", 'epcm', 'didn', 'been', 'co', 'auto', 'equi', 'green', 'state', 'customer', 'andrea', 'we', 'manuf', 'they', 'aaaaaaaaaa', 'sample', 'while', "it's", 'or', 'item', 'material', 'equipment', 'eequipment', 'air', 'care', 'device', 'devices','rough','preparation','contents','preparations','esd', 'system','rubber', 'analysis', 'apparatus', 'arm', 'attachment', 'bath', 'card', 'carrier', 'communication', 'component', 'compound', 'consumer', 'cream', 'development', 'distributor', 'eye', 'file', 'flow', 'foot', 'foot', 'frame', 'framework', 'hand', 'head', 'heavy', 'high', 'house', 'household', 'image', 'induction', 'inspection', 'korean', 'lead', 'level', 'line', 'loss', 'maker', 'massage', 'master', 'medium', 'mixed', 'mother', 'operator', 'photo', 'power', 'saw', 'service', 'side', 'sight', 'sign', 'small', 'software', 'solution', 'speaker', 'station', 'street', 'structure', 'supply', 'support', 'survey', 'sweet', 'synthetic', 'system', 'technology', 'textile', 'tights', 'transfer', 'transformer', 'transmission', 'treatment', 'value', 'vision', 'weight', 'false','flake','crop','crops','mixture','pack','mayu','plant','plants','root','water','based','sauces','accessories','refill']


		self.exist_3wd_li = ['jar', 'way', 'cam', 'dye', 'spa', 'pig', 'mug', 'mat', 'gel', 'soy', 'map', 'eye', 'dog', 'kid', 'zoo', 'gas', 'egg', 'bow', 'cap', 'art', 'toy', 'hat', 'cat', 'die', 'cup', 'gym', 'jam', 'bio', 'cow', 'gun', 'tea', 'car', 'bag', 'led', 'box', 'wig', 'bed', 'pen', 'key', 'pan', 'tag', 'pet', 'lip', 'fur', 'vet', 'pad', 'bus', 'pot', 'ice', 'mop', 'ink', 'gum', 'wax', 'gin']

		#약어
		self.exist_3abv_li = ['api', 'hud', 'dvd', 'atm', 'lan', 'pvc', 'app', 'iot', 'gps', 'usb', 'ssd', 'ram', 'utp', 'msg', 'cpu', 'dna', 'bbq', 'suv', 'hdd', 'pdf', 'vpn', 'cad', 'web', 'mep']

		self.exist_li = self.exist_3wd_li+self.exist_3abv_li






	def df_newline_tab_replace(self, data, columns):
		"""
		개행 및 tab 문자열 -> '' 변경 함수
		Args:
			data=dataframe
			columns=column_name
		Returns:
			dataframe
		"""
		data[columns] = data[columns].apply(lambda x: x.replace('\n', '').replace('\r','').replace('\t','').strip())

		return data


	def final_preprocessing(self, final_result):
		"""
		Buyer ID 별로 매핑된 list 내 MTICODE를 하나씩 하나의 row로 변경해 
		MTICODE 별 데이터프레임으로 반환해주는 함수 
		Args:
			final_result=최종 병합된 MTIcode list가 들어있는 dataframe
		Returns:
			dataframe
		"""	
		seri_to_df = final_result['MTICD'].apply(lambda x: pd.Series(x))
		seri_to_df2 = seri_to_df.stack().reset_index(level=1, drop=True).to_frame('mticode')
		seri_to_df2 = seri_to_df2[seri_to_df2['mticode']!='x']
		r_final_result = final_result.merge(seri_to_df2, left_index=True, right_index=True, how='left')
		r_final_result = r_final_result.reset_index(drop=True)

		
		output = r_final_result[r_final_result['mticode'].notnull()]

		output['대표품목명'] = output['대표품목명'].fillna('')
		output['대한관심품목명'] = output['대한관심품목명'].fillna('')

		output['대표품목명'] = output['대표품목명'].astype(str)
		output['대한관심품목명'] = output['대한관심품목명'].astype(str)


		output['품목명'] = output['대표품목명']+' & '+output['대한관심품목명']
		output['품목명_li'] = output['품목명'].str.split(' & ')
		output['품목명_li'] = output['품목명_li'].apply(lambda x: list(set(x)))
		output['품목명'] = output['품목명_li'].apply(lambda x: ' & '.join(x))
		output.loc[output['품목명'].str.startswith(' &'), '품목명'] = output['품목명'].str[3:]


		output2 = output[['mticode','BUYERID','품목명']]
		output2['품목명'] = output2['품목명'].str.replace(' & ', '')
		output2 = output2.rename(columns={'mticode':'MTICD'})
		output2 = output2.drop_duplicates(subset=['MTICD','BUYERID','품목명'])
		output2 = output2.reset_index(drop=True)
		output2 = output2.rename(columns={'BUYERID':'BUYER_ID','품목명':'UNTY_RPSNT_CMDLT_NAME'})
		
		return output2


	def fill_na(self, data, substitute_word):
		"""
		Null 값 대체 함수
		Args:
			data=dataframe
			substitute_word= 대체어, str
		Returns:
			dataframe
		"""
		data = data.fillna(substitute_word) # ''

		return data



	def pos_noun(self, x):
		"""
		명사형 단어만 추출['NNG','NNB','NNP']
		1글자 이하 제거
		한글 불용어에 없는 단어만 
		Args:
			x=string
		Returns:
			list
		"""

		text=[]
		if isinstance(x, list):
			for i in x:
				if (i[1] in ['NNG','NNB','NNP']) & (len(i[0])>1) & (i[0] not in self.ko_stopwords):
					text.append(i[0])

		return text 



	def itri_mapping_MTICODE(self, data, mti_ko_en, column_name, ko_yn):
		"""
		전처리된 텍스트 데이터에서 해당하는 컬럼의 
		단어에 매칭되는 MTI code 찾아 매핑하는 함수
		Args:
			data=dataframe
			mti_ko_en=MTI_KO_EN.csv
			column_name=컬럼명
			ko_yn = 한글여부(한글이면 1, 영어면 0)
		Returns:
			dataframe

		""" 
		if ko_yn == 1:
			column_name2 = column_name + '_ext_ko_nouns'
			mti_col_name = 'MTICD_DC'
			add_col_nm = '_MTICD'
		elif ko_yn == 0:
			column_name2 = column_name + '_token_lemma'
			mti_col_name = 'MTICD_ENG_DC'
			add_col_nm = '_EN_MTICD'
		else:
			print('[ko_yn] Enter only Korean(1) or English(0).')

		results = []
		for idx, i in enumerate(tqdm(data[column_name2])):

			if (isinstance(i, list))&(len(i)>0):
				i_li = i

				if (len(i_li)<= 3)&(i_li[0]!=''):
					input_values = [None, None, None]

					for idx in range(len(i_li)):
						input_values[idx] = i_li[idx]

					input_value1, input_value2, input_value3 = input_values[0], input_values[1], input_values[2]
					if ko_yn == 1:
						mti_co_li = self.output_en_and(input_value1, input_value2, input_value3, mti_ko_en, mti_col_name)  
					elif ko_yn == 0:
						mti_co_li = self.output_en_and(input_value1, input_value2, input_value3, mti_ko_en, mti_col_name) 


				elif (len(i_li)> 3)&(i_li[0]!=''):
					i_li = list(map(lambda x: x.strip().replace(' ',''), i_li))

	#				 input_value1, input_value2, input_value3 = '|'.join(i_li), None, None
					if ko_yn == 1:
						mti_co_li = self.output_en_or(i_li, mti_ko_en, mti_col_name)  
					elif ko_yn == 0:
						mti_co_li = self.output_en_or(i_li, mti_ko_en, mti_col_name)

				results.append(mti_co_li)

			else:
				mti_co_li = []
				results.append(mti_co_li) 
		res_col_name = column_name + add_col_nm					 
		data[res_col_name] = results

		return data

	def english_extract2(self, data_eng, column_name):
		"""
		영어 추출 함수
		Args:
			data_eng=dataframe
			column_name=컬럼명
		Returns:
			dataframe	
		"""

		data_eng[f"{column_name}_token"] = data_eng[column_name].apply(lambda x: re.sub('〮|,|/|\.|-|&|\(|\)|，|;', ', ', x))
		data_eng[f"{column_name}_token"] = data_eng[f"{column_name}_token"].apply(lambda x: re.sub('[^a-zA-Z\s]+', '', x).strip())

		return data_eng

	def tokenization_intritem(self, data_eng):
		"""
		tokenize 함수
		Args:
			data_eng=dataframe
		Returns:
			dataframe  
		"""
		data_eng["대표품목명_token"] = data_eng["대표품목명_token"].apply(lambda x: nltk.word_tokenize(x))
		data_eng["대한관심품목명_token"] = data_eng["대한관심품목명_token"].apply(lambda x: nltk.word_tokenize(x))   


		return data_eng

	def hap_len_drop(self, data_eng, column_name):
		"""
		hap의 길이가 1 이상인 행 추출 함수
		Args:
			data_eng=dataframe
			column_name=컬럼명
		Returns:
			dataframe  
		"""
		column_name2 = column_name + '_token'

		data_eng[f'{column_name2}_len'] = data_eng[column_name2].apply(lambda x: len(x))
		data_eng = data_eng[data_eng[f'{column_name2}_len'] >= 1]
		data_eng = data_eng.reset_index(drop=True)
		data_eng = data_eng.drop(f'{column_name2}_len', axis=1)

		return data_eng


	# 위에 동일한 함수에 세자리 제외 조건 추가 
	# 대한, 대표 - 3글자이하 제거 but, 3글자는 남길 단어 리스트에 존재시 제거안함
	def en_noun_ext(self, data_eng, column_name):
		"""
		품사 태깅으로 명사가 아닌 것들에 대해서 lemmatization 후 명사 추출 함수
		Args:
			data_eng=dataframe
			column_name=컬럼명
		Returns:
			dataframe

		"""
		data_eng[f'{column_name}_lemma'] = data_eng[column_name]
		for i in tqdm(range(data_eng.shape[0])):
			li = []
			for word in data_eng.loc[i:i, column_name].values[0]: # 리스트
				for w, t in nltk.pos_tag([word]): # 리스트 내에서 하나씩
					w = nltk.WordNetLemmatizer().lemmatize(w, pos='n')
					if t not in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
						if (nltk.pos_tag([w])[0][1] in ["NN", "NNS", "NNP", "NNPS", "JJ"])&((len(w)>3)|(w in self.exist_li)):
							li.append(w)

					elif (t in ["NN", "NNS", "NNP", "NNPS", "JJ"])&((len(w)>3)|(w in self.exist_li)):
						li.append(w)
			data_eng.loc[i:i, f'{column_name}_lemma'].values[0] = li

		data_eng[f'{column_name}_lemma'] = data_eng[f'{column_name}_lemma'].apply(lambda x: [i for i in x if i not in self.en_stopwords])

		return data_eng




	def data_ko_en_merge(self, data, kor_result, eng_result):
		"""
		한글처리결과, 영어처리결과를 맨 처음 data 형태에 left join
		Args:
			data=preprocessing 함수의 input data dataframe
			kor_result=한글 매핑 결과 dataframe
			eng_result=영어 매핑 결과 dataframe
		Returns:
			dataframe 
		"""
		kor_result_sub = kor_result[["BUYERID", "MTICD"]]
		kor_result_sub.columns = ["BUYERID", "KOR_MTICD"]
		eng_result_sub = eng_result[["BUYERID", "EN_MTICD"]]
		eng_result_sub.columns = ["BUYERID", "ENG_MTICD"]	 

		final_result = pd.merge(data, kor_result_sub, how='left', on='BUYERID')
		final_result = pd.merge(final_result, eng_result_sub, how='left', on='BUYERID')

		return final_result


	def assign_mticode(self, data):
		"""
		조건별 MTI CODE 추출 함수 
		MTI CODE 앞에서 4자리의 빈도가 3이상인 값들 중 MTI CODE 개수가 5미만은 다 나오고 5이상은 최대빈도 3개만 추출
		Args:
			data=dataframe
		Returns:
			dataframe	
		"""
		data['hscd4_li'] = 'x'
		data['MTI4_Counting_Result'] = 'x' 


		for i in tqdm(range(data.shape[0])):
			li = []
			if (data.loc[i:i, "MTICD"].values[0] == None):
				pass
			elif (data.loc[i:i, "MTICD"].values[0] != None):
				if (len(data.loc[i:i, "MTICD"].values[0]) <= 4):
	#				 data.loc[i:i, "MTI4_Counting_Result"].values[0] = data.loc[i:i, "final_MTICD"].values[0]
					li.extend(data.loc[i:i, "MTICD"].values[0])
				else:
					four_li = [num[:4] for num in data.loc[i:i, "MTICD"].values[0]]
					counting_res = Counter(four_li)

					data.loc[i:i, "hscd4_li"].values[0] = counting_res
	#				 four_li_li = [i[0] for i in Counter(four_li).items() if i[1] >= 2]
					four_li_li = [i[0] if (len(i[0]) < 5) else i[0][:4] for i in counting_res.most_common(10) if (i[1] > 3)]  

					for num in data.loc[i:i, "MTICD"].values[0]:
						for f in four_li_li:
							if str(num).startswith(str(f)):
								li.append(num)
				data.loc[i:i, "MTI4_Counting_Result"].values[0] = li

		return data

	def lower_and_strip(self, data, column_name):
		"""
		소문자 처리 및 양쪽 공백 제거 함수
		Args:
			data=dataframe
			column_name=컬럼명
		Returns:
			dataframe

		""" 
		data[column_name] = data[column_name].apply(lambda x: x.lower().strip())

		return data


	def preprocessing_intriitem(self, data, mti_ko_en): 
		"""
		대표 품목명, 대한관심품목명 컬럼에서 한글과 영어 텍스트 데이터 전처리 함수
		Args:
			data=dataframe
			mti_ko_en=MTI_KO_EN.csv
		Returns:
			dataframe

		"""	 
		date = time.strftime('%Y%m%d')

		# ================================== 한글
		data_ko = self.fill_na(data, '')

		data_ko['대표품목명_ext_ko'] = data_ko['대표품목명'].progress_apply(lambda x : self.korean_extract(x))
		data_ko['대한관심품목명_ext_ko'] = data_ko['대한관심품목명'].progress_apply(lambda x : self.korean_extract(x))

		data_ko = self.df_newline_tab_replace(data_ko, '대표품목명')
		data_ko = self.df_newline_tab_replace(data_ko, '대한관심품목명')

		data_ko = data_ko[(data_ko['대표품목명_ext_ko'].str.strip()!='')|(data_ko['대한관심품목명_ext_ko'].str.strip()!='')]# 두 컬럼 모두 NUlls이면 drop
		data_ko = data_ko.reset_index(drop=True)
		tagging = self.morpheme_analysis_todict(data_ko)
		
		#data_ko['대표품목명_ext_ko_nouns'] = data_ko['대표품목명_ext_ko'].progress_apply(self.morpheme_analysis)
		#data_ko['대한관심품목명_ext_ko_nouns'] = data_ko['대한관심품목명_ext_ko'].progress_apply(self.morpheme_analysis)

		data_ko['대표품목명_ext_ko_nouns'] = data_ko['대표품목명_ext_ko'].apply(lambda x : tagging.get(x))
		data_ko['대한관심품목명_ext_ko_nouns'] = data_ko['대한관심품목명_ext_ko'].apply(lambda x : tagging.get(x))

		data_ko['대표품목명_ext_ko_nouns'] = data_ko['대표품목명_ext_ko_nouns'].progress_apply(self.pos_noun)  # extract noun 
		data_ko['대한관심품목명_ext_ko_nouns'] = data_ko['대한관심품목명_ext_ko_nouns'].progress_apply(self.pos_noun)  # extract noun 


		data_ko_map = self.itri_mapping_MTICODE(data_ko, mti_ko_en, '대표품목명', ko_yn=1)
		ko_result = self.itri_mapping_MTICODE(data_ko, mti_ko_en, '대한관심품목명', ko_yn=1)
		ko_result['MTICD'] = ko_result['대표품목명_MTICD']+ko_result['대한관심품목명_MTICD']

		#with open(f'preprocessed_01_ko_{date}.pkl', 'wb') as f:
		with open(f'preprocessed_01_ko.pkl', 'wb') as f:
			pickle.dump(ko_result, f)

		# ================================== 영어
		data_en = self.fill_na(data, '')

		data_en = self.lower_and_strip(data_en, "대표품목명")
		data_en = self.lower_and_strip(data_en, "대한관심품목명")

		data_en = self.english_extract2(data_en, '대표품목명')
		data_en = self.english_extract2(data_en, '대한관심품목명')

		data_en = data_en[(data_en['대표품목명_token'].str.strip()!='')|(data_en['대한관심품목명_token'].str.strip()!='')]
		data_en = data_en.reset_index(drop=True)

		data_en = self.tokenization_intritem(data_en)
		data_en = self.hap_len_drop(data_en, '대표품목명')
		data_en = self.hap_len_drop(data_en, '대한관심품목명')


		data_en['대표품목명_token'] = data_en['대표품목명_token'].apply(lambda x: [i for i in x if i not in self.en_stopwords])
		data_en['대한관심품목명_token'] = data_en['대한관심품목명_token'].apply(lambda x: [i for i in x if i not in self.en_stopwords])

		data_en = self.en_noun_ext(data_en, '대표품목명_token')
		data_en = self.en_noun_ext(data_en, '대한관심품목명_token')


		word_list = list(set(list(chain(*mti_ko_en['MTICD_ENG_DC'].str.split(', | ').tolist()))))			# 영어는 HS_DESC_EN 단어만 있는 경우로 한정
		uni_word_list = [i for i in word_list if len(i)>1]

		data_en['대표품목명_token_lemma'] = data_en['대표품목명_token_lemma'].apply(lambda x: [i for i in x if i in uni_word_list])
		data_en['대한관심품목명_token_lemma'] = data_en['대한관심품목명_token_lemma'].apply(lambda x: [i for i in x if i in uni_word_list])

		data_en = data_en[(data_en['대표품목명_token_lemma'].str.len()!=0)&(data_en['대한관심품목명_token_lemma'].str.len()!=0)] 

		data_en_map = self.itri_mapping_MTICODE(data_en, mti_ko_en, '대표품목명', ko_yn=0)
		en_result = self.itri_mapping_MTICODE(data_en, mti_ko_en, '대한관심품목명', ko_yn=0)

		en_result['EN_MTICD'] = en_result['대표품목명_EN_MTICD']+en_result['대한관심품목명_EN_MTICD']

		#with open(f'preprocessed_01_en_{date}.pkl', 'wb') as f:
		with open(f'preprocessed_01_en.pkl','wb') as f:
			pickle.dump(en_result, f)

		# ================================== 한글 영어 병합
		result = self.data_ko_en_merge(data, ko_result, en_result)

		result['KOR_MTICD'] = result['KOR_MTICD'].fillna('').apply(list)
		result['ENG_MTICD'] = result['ENG_MTICD'].fillna('').apply(list)
		result['MTICD'] = result['KOR_MTICD'] + result['ENG_MTICD']

		res = result[['BUYERID','대표품목명','대한관심품목명','MTICD']]


		return res  


	def drop_null(self, data, subset_colname):
		"""
		Drop null or blank values 함수
		Args:
			data=dataframe
			subset_colname=str
		Returns:
			dataframe
		"""						
		data = data.dropna(subset=[subset_colname], axis = 0)
		data = data[data[subset_colname].str.strip()!='']
		data = data.reset_index(drop=True)						 
		return data




	def korean_extract(self, x):
		"""
		한글 추출 함수
		Args:
			x=string
		Returns:
			string
		"""						 
		p = re.compile('[^가-힣\s]*')
		x = p.sub('', x)
		x = x.strip()
		return x

	def morpheme_analysis(self, x):
		"""
		형태소 분석 함수
		Args:
			x=string
		Returns:
			list(tuple)
			ex) [(레이저, NNP), (소자, NNP)]
		"""

		dics = '/home/bpetl/BPETL/analysis/bin/model/buyer_mti/custom_dictionary2.txt'

		ko_words=[]
		if x.strip()!='':
			ko = Komoran(userdic=dics)
			ko_res_li = ko.pos(x)
			for val in ko_res_li:
				if val[0] not in self.ko_stopwords:
					ko_words.append(val)

		return ko_words						 
	

	def morpheme_analysis_todict(self, data):
		"""
		형태소 분석 결과 저장 및 업데이트 후 저장된 딕셔너리 결과 LOAD
		Args:
			data:dataframe
		Returns:
			dictionary
			ex) '레이저 소자': [(레이저, NNP), (소자, NNP)]
		"""
		dics = '/home/bpetl/BPETL/analysis/bin/model/buyer_mti/custom_dictionary2.txt'
		ko = Komoran(userdic=dics)
		date = time.strftime('%Y%m%d')
		
		try:
			morph_file_name = sorted(glob.glob("/home/bpetl/BPETL/analysis/bin/model/buyer_mti/morph_dict.pkl"), reverse=True)[0]
		
			with open(morph_file_name, 'rb') as f:
				tagging = pickle.load(f)
		except IndexError:
			tagging = dict()

		daepyo = data['대표품목명_ext_ko'].tolist()
		daehan = data['대한관심품목명_ext_ko'].tolist()

		dae = daepyo + daehan
		dae_li = list(set(dae))

		for i in tqdm(dae_li):
			if tagging.get(i):
				pass
			else:
				ko_res_li = ko.pos(i)

				ko_res_li_rp = []
				for val in ko_res_li:
					if val[0] not in self.ko_stopwords:
						ko_res_li_rp.append(val)
						
				tagging[i] = ko_res_li_rp
		
		with open(f'/home/bpetl/BPETL/analysis/bin/model/buyer_mti/morph_dict.pkl', 'wb') as f:
			pickle.dump(tagging, f)

		return tagging


	def contains_func(self, pattern, data, column_name):
		"""
		MTI_KO_EN.csv 내 DESC 컬럼에서 패턴에 해당하는 데이터 추출 함수
		Args:
			pattern=DESC내 찾고자하는 문자열의 패턴
			data=dataframe(MTI_KO_EN.csv)
			column_name=DESC 컬럼명
		Returns:
			dataframe

		"""						 
		return data[data[column_name].str.contains(pattern)]



	def output_en_and(self, input_value1, input_value2, input_value3, data, column_name):
		"""
		contains_func 실행해 mticode list 추출 함수
		Args:
			input_value1, input_value2, input_value3= 조회 조건 키워드
			data=dataframe(MTI_KO_EN.csv)
			column_name=DESC 컬럼명
		Returns:
			list : MITCODE list

		""" 

		search_words_list = [input_value1, input_value2, input_value3]
		pattern1 = '^'+''.join(fr'(?=.*( |,){x}( |,))' for x in search_words_list if x is not None) 


		result = self.contains_func(pattern1, data, column_name = column_name) 

		result = result[['MTICD','MTICD_NAME']]
		result_mticd = result['MTICD'].tolist()

		return result_mticd 

	def output_en_or(self, kwd_list, data, column_name):
		"""
		contains_func 실행해 mticode list 추출 함수
		Args:
			kwd_list=조회 할 키워드 리스트
			data=dataframe(MTI_KO_EN.csv)
			column_name=DESC 컬럼명
		Returns:
			list : MITCODE list

		""" 
		add_txt = '( |,)'
		search_words_list = [add_txt + i + add_txt for i in kwd_list if i is not None]
		pattern1 = '^'+''.join(fr'(?=.*({"|".join(search_words_list)}))')



		result = self.contains_func(pattern1, data, column_name = column_name) 

		result = result[['MTICD','MTICD_NAME']]
		result_mticd = result['MTICD'].tolist()

		return result_mticd 

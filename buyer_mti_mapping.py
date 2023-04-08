import pandas as pd
import numpy as np
import time
import csv
import sys
import re
import os
import pickle
import pyodbc

from buyer_mti_analysis import PreProcess
import Mobigen.Common.Log_PY3 as Log;Log.Init()

class AnalysisBuyerMTI:
	def __init__(self, args=None):
		self.buyer_sql = """
					SELECT DISTINCT BUYERID, REPITEMNAME AS 대표품목명, INTRITEMDESC AS 대한관심품목명
					FROM BPUSER.TB_CM_BUYER A 
					WHERE A.COSTATE ='01'
					AND (A.INTRITEMDESC IS NOT NULL OR A.REPITEMNAME IS NOT NULL)
					-- AND ((A.REGDATE >= (SELECT max(to_date(B.DATA_REG_DT,'YYYYMMDD HH24MISS'))-2 FROM BPUSER.TB_BUYER_MTI_MAPNG_CD B)) 
					--	OR (A.EDTDATE >= (SELECT max(to_date(B.DATA_REG_DT,'YYYYMMDD HH24MISS'))-2 FROM BPUSER.TB_BUYER_MTI_MAPNG_CD B)))
					
				   """
	
		#self.mti_sql = """
		#			SELECT *
		#			FROM BPUSER.TB_MTICD_INFO_BUYER_MAPNG
		#			"""
		self.date = time.strftime('%Y%m%d')
	
		
	def getData(self):
		"""
		TIBERO DB 수집
		SQL 실행 함수
		Args:
			sql=SQL
		Returns:
			dataframe
		"""
		conn = pyodbc.connect('DSN=BP_DB2;UID=bpuser;PWD=kotrabp')
		conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
		conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
		conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le')
		conn.setencoding(encoding='utf-8')

		
		data= pd.read_sql_query(self.buyer_sql, conn)
		#mti_ko_en= pd.read_sql_query(self.mti_sql, conn)
		mti_ko_en = pd.read_csv('/home/bpetl/BPETL/analysis/bin/model/buyer_mti/TB_MTICD_INFO_BUYER_MAPNG.csv', dtype={'MTICD':'string'})
		conn.close()

		return data, mti_ko_en
	

	
	def analysis(self):
		__LOG__.Trace("========================== Data Load ==========================") 
		data, mti_ko_en = self.getData()
		print(data.shape)
		print(mti_ko_en.shape)

		__LOG__.Trace("========================== preprocessing ==========================") 
		prep = PreProcess()
		tot_intri_result = prep.preprocessing_intriitem(data, mti_ko_en)

		__LOG__.Trace("============================ MTICODE mapping ==========================") 
		tot_intri_result = prep.assign_mticode(tot_intri_result)  

		__LOG__.Trace("========================== preprocessing Done. ==========================") 

		dt = tot_intri_result[['BUYERID','대표품목명','대한관심품목명','MTI4_Counting_Result']]
		dt = dt.rename(columns={'MTI4_Counting_Result':'MTICD'})

		dt = dt[['BUYERID','대표품목명','대한관심품목명','MTICD']]
		dt = dt[dt['MTICD'].str.len()!=0]
		dt = dt.reset_index(drop=True)


		__LOG__.Trace("========================== postprocessing ==========================") 
		output = prep.final_preprocessing(dt)

		__LOG__.Trace("========================== Final Done. ==========================")	
		
		return output
	
	
	def run(self):

		
		df = self.analysis()
		data_reg_dt = time.strftime('%Y%m%d%H%M%S')
		df['DATA_REG_DT'] = data_reg_dt
		
		return {'TB_BUYER_MTI_MAPNG_CD':df}
		
		
	def save(self, save_file, df):
		
		save_file_nm = f'/home/bpetl/BPETL/analysis/bin/model/buyer_mti/{save_file}'
		
		df['TB_BUYER_MTI_MAPNG_CD'].to_csv(save_file_nm, sep = ',', quotechar='"', quoting=csv.QUOTE_ALL, line_terminator=False, index=False, header=True, encoding='utf-8-sig')
		
		
	def make(self, save_path='./', arg1=None):
		
		result = self.run()
		#data_reg_dt = time.strftime('%Y%m%d%H%M%S')
		#result['DATA_REG_DT'] = data_reg_dt
		save_file = 'Buyer_MTICODE_Mapping_Result_toDB.csv'
		self.save(save_file, result)
		
		return result
	


if __name__ == '__main__':
	obj = AnalysisBuyerMTI()
	
	result = obj.make()
	
	save_file = 'Buyer_MTICODE_Mapping_Result_toDB.csv'
	obj.save(save_file, result)

'''
Loaders for data in mongodb database(stocks.ml).
'''
import os
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
from typing import Optional, Union, List
from ..utils import load_secrets


def create_connection():
    secrets = load_secrets()
    client = MongoClient('mongodb://mongo:27017/',
                          username=secrets['mongodb_adminusername'],
                          password=secrets['mongodb_adminpassword'])
    db = client.ml_investment
    return db




def get_group_top_pipeline(group_col: str, sorted_col: str, top: int):
    pipeline = [
        { 
            '$sort': {
                group_col: pymongo.DESCENDING,
                sorted_col: pymongo.DESCENDING 
            } 
        },
        { 
            '$group': {
                '_id': '$'+group_col,
                'docs': {'$push': '$$ROOT'},
            }
        },
        {
            '$project': {
                'docs': { 
                    '$slice': ['$docs', top]
                }
            }
        },
        { 
            "$unwind": "$docs"
        },
        {
            "$replaceRoot": { 
                "newRoot": "$docs"
            }
        },
    ]

    return pipeline



class SF1BaseData: 
    '''
    Load base information about company(like sector, industry etc)
    '''
    def __init__(self, db: Optional[pymongo.database.Database]=None):
        '''
        Parameters
        ----------
        db:
            pymongo database to load data from. 
            If None, than connection will be created from secrets file:
            fields ``mongodb_adminusername`` and ``mongodb_adminpassword`` 
            at `~/.ml_investment/secrets.json`
        '''
        if db is None:
            db = create_connection()
        self.db = db


    def load(self, index: Optional[List[str]]=None) -> pd.DataFrame:
        '''
        Parameters
        ----------
        index:
            list of ticker to load data for, i.e. ``['AAPL', 'TSLA']`` 
            OR ``None`` (loading for all possible tickers)
        Returns
        -------
        ``pd.DataFrame`` 
            base companies information
        '''
        if index is not None:
            cursor = self.db["sf1_base"].find({"ticker": {'$in': index} })
        else:
            cursor = self.db["sf1_base"].find()

        result = [x for x in cursor]
        if len(result) == 0:
            return None

        result = pd.DataFrame(result)

        return result



class SF1QuarterlyData: 
    '''
    Loader for quartely fundamental information about
    companies(debt, revenue etc)
    '''
    def __init__(self,
                 db: Optional[pymongo.database.Database]=None,
                 quarter_count: Optional[int]=None,
                 dimension: Optional[str]='ARQ'):
        '''
        Parameters
        ----------
        db:
            pymongo database to load data from. 
            If None, than connection will be created from secrets file:
            fields ``mongodb_adminusername`` and ``mongodb_adminpassword`` 
            at `~/.ml_investment/secrets.json`
        quarter_count:
            maximum number of last quarters to return. 
            Resulted number may be less due to short history in some companies
        dimension: 
            one of ``['MRY', 'MRT', 'MRQ', 'ARY', 'ART', 'ARQ']``.
            SF1 dataset-based parameter
        '''
        if db is None:
            db = create_connection()
        
        if quarter_count is None:
            quarter_count = 200

        self.db = db
        self.quarter_count = quarter_count
        self.dimension = dimension


    def load(self, index: List[str]) -> pd.DataFrame:
        '''    
        Parameters
        ----------
        index:
            list of tickers to load data for, i.e. ``['AAPL', 'TSLA']``
           
        Returns
        -------
        ``pd.DataFrame``
            quarterly information about companies
        '''
        pipeline = [{'$match': 
                        {'ticker': {'$in': index},
                         'dimension': self.dimension}
                    }]

        pipeline.extend(get_group_top_pipeline(group_col='ticker',
                                               sorted_col='date',
                                               top=self.quarter_count))
        result = [x for x in self.db["sf1_quarterly"].aggregate(pipeline)]
        if len(result) == 0:
            return None

        result = pd.DataFrame(result)
        result['date'] = result['date'].apply(lambda x: np.datetime64(x, 'ms'))

        return result
      

        
class SF1DailyData():
    '''
    Load daily information about company(marketcap, pe etc)
    '''
    def __init__(self,
                 db: Optional[pymongo.database.Database]=None,
                 days_count: Optional[int]=None):
        '''
        Parameters
        ----------
        db:
            pymongo database to load data from. 
            If None, than connection will be created from secrets file:
            fields ``mongodb_adminusername`` and ``mongodb_adminpassword`` 
            at `~/.ml_investment/secrets.json`
        days_count:
            maximum number of last days to return. 
            Resulted number may be less due to short history in some companies
        '''
        if db is None:
            db = create_connection()
        
        if days_count is None:
            days_count = int(1e5)    

        self.db = db
        self.days_count = days_count


    def load(self, index: List[str]) -> pd.DataFrame:
        '''    
        Parameters
        ----------
        index:
            list of ticker to load data for, i.e. ``['AAPL', 'TSLA']`` 
             
        Returns
        -------
        ``pd.DataFrame``
            daily information about companies
        '''                        
        pipeline = [{'$match': 
                        {'ticker': {'$in': index}}
                    }]
        
        pipeline.extend(get_group_top_pipeline(group_col='ticker',
                                               sorted_col='date',
                                               top=self.days_count))

        result = [x for x in self.db["sf1_daily"].aggregate(pipeline)]
        if len(result) == 0:
            return None

        result = pd.DataFrame(result)
        result['date'] = result['date'].apply(lambda x: np.datetime64(x, 'ms'))

        return result



class QuandlCommoditiesData:
    '''
    Loader for commodities price information. 
    '''
    def __init__(self, db: Optional[pymongo.database.Database]=None):
        '''
        db:
            pymongo database to load data from. 
            If None, than connection will be created from secrets file:
            fields ``mongodb_adminusername`` and ``mongodb_adminpassword`` 
            at `~/.ml_investment/secrets.json`
        '''
        if db is None:
            db = create_connection()
        self.db = db
       

    def load(self, index: List[str]) -> pd.DataFrame:
        '''
        Load time-series information about commodity price
        
        Parameters
        ----------
        index:
            list of commodities codes to load data for, i.e. 
            ``['LBMA/GOLD', 'JOHNMATT/PALL']``
            
        Returns
        -------
        ``pd.DataFrame``
            time series price information
        '''  
        index = [x.replace('/', '_') for x in index]
        cursor = self.db["quandl_commodities"].find({'commodity_code': 
                                                        {'$in': index}})
        cursor = cursor.sort('date', pymongo.DESCENDING)
        result = [x for x in cursor]
        result = pd.DataFrame(result)
        result['date'] = result['date'].apply(lambda x: np.datetime64(x, 'ms'))
        
        return result



class DailyBarsData:
    '''
    Loader for daywise price bars.
    '''
    def __init__(self,
                 db: Optional[pymongo.database.Database]=None,
                 collection_name: str='daily_bars',
                 days_count: Optional[int]=None):
        '''
        Parameters
        ----------
        db:
            pymongo database to load data from. 
            If None, than connection will be created from secrets file:
            fields ``mongodb_adminusername`` and ``mongodb_adminpassword`` 
            at `~/.ml_investment/secrets.json`
        collection_name:
            name of collection containing daily bars information
        days_count:
            maximum number of last last days to return. 
            Resulted number may be less due to short history in some companies
        '''
        if db is None:
            db = create_connection()
        
        if days_count is None:
            days_count = int(1e5)
        
        self.db = db
        self.collection_name = collection_name
        self.days_count = days_count


    def load(self, index: List[str]) -> pd.DataFrame:
        '''
        Load daily price bars
        
        Parameters
        ----------
        index:
            list of tickers to load data for, i.e. ``['AAPL', 'TSLA']``
            
        Returns
        -------
        ``pd.DataFrame``
            daily price bars
        '''                        
        pipeline = [{'$match': 
                        {'ticker': {'$in': index}}
                    }]
        
        pipeline.extend(get_group_top_pipeline(group_col='ticker',
                                               sorted_col='date',
                                               top=self.days_count))

        result = [x for x in self.db[self.collection_name].aggregate(pipeline)]
        if len(result) == 0:
            return None

        result = pd.DataFrame(result)
        result['date'] = result['date'].apply(lambda x: np.datetime64(x, 'ms'))

        return result









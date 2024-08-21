import os 
import pandas as pd


PROJECT_DIR = "/data/rwchain-all/round2"
BEH_DIR = os.path.join(PROJECT_DIR, "rwchain-beh/data")
EEG_DIR = os.path.join(PROJECT_DIR, "rwchain-eeg")
ALL_BEH_DIR = os.path.join(PROJECT_DIR, 'rwchain-beh', 'combined')




class Dataset:
    # class variables
    PROJECT_DIR = "/data/rwchain-all/round2"
    BEH_DIR = os.path.join(PROJECT_DIR, "rwchain-beh/data")
    EEG_DIR = os.path.join(PROJECT_DIR, "rwchain-eeg")
    ALL_BEH_DIR = os.path.join(PROJECT_DIR, 'rwchain-beh', 'combined')
    CODE_DIR = "/home/jenny/evidence-chain/"

    def __init__(self, fig_dir, stimdur): 
        self.fig_dir = fig_dir
        self.stimdur = stimdur
    
    @property
    def metadata(self, fname =  'all_df_concat.pkl'):
        '''get the concatenated df without needing to write parentheis'''
        df = pd.read_pickle(os.path.join(ALL_BEH_DIR, fname))
        # organize some columsn
        df['key'][df['key'] == '[5]'] = 1
        df['key'][df['key'] == '[3]'] = 1
        df['key'][df['key'] == '[2]'] = 0
        df['cumsum'] = df['sequence_clean'].apply(lambda x: [sum(x[:i+1]) for i in range(len(x))])
        # get rid of a ;pw acc subject
        df = df[df['sid']!='s108']
        return df
    
    @metadata.setter
    def metadata(self, value:)
    
    def get_dataset(self):
        df = get_metadata
        if stimdur == '100':
    df = df[df['stimDur'] == 0.1]
if stimdur == '250':
    df = df[df['stimDur'] == 0.25]
        
        
        
    @classmethod 
    def list_of_subjects(cls):
        list_of_subj = os.listdir(cls.BEH_DIR)
        list_of_subj.sort() 
        return list_of_subj

    @classmethod
    def change_data_directory(cls, project_dir, beh_dir, eeg_dir, all_beh_dir):
        cls.PROJECT_DIR = project_dir
        cls.BEH_DIR = beh_dir
        cls.EEG_DIR = eeg_dir
        cls.ALL_BEH_DIR = all_beh_dir





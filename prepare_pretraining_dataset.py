import numpy as np
import random
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
from config_pretrain import sampling_config,SEED,is_valid_pos,custom_distance
import random
import threading
from tqdm import tqdm
from pathlib import Path


def main(city):
    np.random.seed(SEED)
    random.seed(SEED)
    datasets = []
    sampled_datas = []
    outputfile = Path(sampling_config.SAMPLE_ROOT)/sampling_config.FILETAG/(city+'.csv')
    if outputfile.exists():
        return
    for src in sampling_config.SRCS:
        dataset_table_path = Path(sampling_config.SAMPLE_ROOT)/src/(city+'.csv')
        if dataset_table_path.exists():
            dataset=pd.read_csv(dataset_table_path,index_col=0)
        else:
            dataset = pd.DataFrame()
            for subsrc in src.split("_"):
                js_path = sampling_config.SRC_PREFIX+subsrc+'_'+city+'.json'
                if not Path(js_path).exists():
                    print(f'missing {js_path}')
                    continue
                dataset = pd.concat([dataset,pd.read_json(js_path)],ignore_index=True)
            assert not dataset[['latitude','longitude']].isnull().any().any()
            dataset=dataset.fillna(' ')
            dataset_table_path.parent.mkdir(exist_ok=True,parents=True)
            dataset.to_csv(str(dataset_table_path))
        datasets.append(dataset)
    dataset_inds = list(range(len(datasets)))

    print(f'start to generate data on {city} with RAW_NEIGHBORHOOD_RADIUS={sampling_config.RAW_NEIGHBORHOOD_RADIUS}, PAIRED_DATASET_SIZE_PER_CITY={sampling_config.PAIRED_DATASET_SIZE_PER_CITY}!')
    tqdm_bar = tqdm(total=sampling_config.PAIRED_DATASET_SIZE_PER_CITY)
    while len(sampled_datas)<sampling_config.PAIRED_DATASET_SIZE_PER_CITY:
        random.shuffle(dataset_inds)
        ind1,ind2 = dataset_inds[:2]

        # find nearest node on two randomly selected dataset
        entity1 = datasets[ind1].sample(n=1)
        distances = datasets[ind2].apply(lambda row: custom_distance([row['latitude'], row['longitude']], [entity1.latitude.iloc[0], entity1.longitude.iloc[0]]), axis=1)
        entity2 = datasets[ind2][datasets[ind2].index == distances.idxmin()]

        # check if entities are valid
        if sampling_config.USE_VALID_POS and not (is_valid_pos(city,lat=entity1.latitude.iloc[0],lon=entity1.longitude.iloc[0]) and
                                                  is_valid_pos(city,lat=entity2.latitude.iloc[0],lon=entity2.longitude.iloc[0])):
            continue
        else:
            tqdm_bar.update(1)

        #find ordered neighbors where the distance less than sampling_config.RAW_NEIGHBORHOOD_RADIUS
        datasets[ind1]['distance'] = datasets[ind1].apply(lambda row: custom_distance([row['latitude'], row['longitude']], [entity1.latitude.iloc[0], entity1.longitude.iloc[0]]), axis=1)
        datasets[ind2]['distance'] = datasets[ind2].apply(lambda row: custom_distance([row['latitude'], row['longitude']], [entity2.latitude.iloc[0], entity2.longitude.iloc[0]]), axis=1)

        entity1_neighbors = datasets[ind1][(datasets[ind1].distance < sampling_config.RAW_NEIGHBORHOOD_RADIUS) & (datasets[ind1].index != entity1.index[0])].sort_values(by='distance')
        entity2_neighbors = datasets[ind2][(datasets[ind2].distance < sampling_config.RAW_NEIGHBORHOOD_RADIUS) & (datasets[ind2].index != entity2.index[0])].sort_values(by='distance')

        sampled_datas.append((entity1.index.to_list()[0],entity1_neighbors.index.to_list(),entity1_neighbors.distance.to_list(),sampling_config.SRCS[ind1],
                            entity2.index.to_list()[0],entity2_neighbors.index.to_list(),entity2_neighbors.distance.to_list(),sampling_config.SRCS[ind2],
                            distances.loc[distances.idxmin()]))
        datasets[ind1] = datasets[ind1].drop("distance", axis=1)
        datasets[ind2] = datasets[ind2].drop("distance", axis=1)

    # save dataset
    outputfile.parent.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(sampled_datas,columns=['entity1_ind','entity1_neighbors_ind','entity1_neighbors_dist','entity1_dataset_name',
                                        'entity2_ind','entity2_neighbors_ind','entity2_neighbors_dist','entity2_dataset_name',
                                        'distance_between_entity1_entity2']).to_csv(outputfile)
        
if __name__=='__main__':
    for city in sampling_config.CITYS:
        main(city)
    # thread_queue = []
    # for city in sampling_config.CITYS:
    #     thread = threading.Thread(target=main, args=(city,))
    #     thread.start()
    # for thread in thread_queue:
    #     thread.join()
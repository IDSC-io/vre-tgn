import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
import time
from sklearn.preprocessing import MinMaxScaler


def preprocess(data_name):
  logger.info("Loading interaction and label data...")
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  logger.info("Loading interation and label data succeeded.")
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}_node_interactions.csv'.format(data_name)
  PATH_NODE_FEAT = './data/{}_node_features.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  # %%

  max_idx = max(new_df.u.max(), new_df.i.max())
  try:
    logger.info("Trying to load graph node features...")
    node_feat = pd.read_csv(PATH_NODE_FEAT)
    node_feat = pd.DataFrame(MinMaxScaler().fit_transform(node_feat.values), columns=node_feat.columns, index=node_feat.index).to_numpy()
    # the indices of the entities start at 1, so we need one more element for the non-existent 0 element (i.e. ml_reddit_df["u"].min() == 1)
    node_feat = np.vstack([node_feat, np.zeros([max_idx + 1 - node_feat.shape[0], node_feat.shape[1]])])
    logger.info("Loading node features succeeded.")
  except Exception as e:
    logger.info("Loading node features failed, loading zero matrix instead...")
    logger.info(str(e))

    node_feat = np.zeros((max_idx + 1, 172))

  # %%

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, node_feat)

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
  parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit or your own)',
                      default='wikipedia')
  parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
  fh.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.WARN)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(ch)
  logger.info(args)

  logger.info(f"Preprocessing {args.data}...")
  run(args.data, bipartite=args.bipartite)
  logger.info(f"Preprocessing {args.data} data succeeded.")
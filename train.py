from model import BinaryClassification
from dataset import IncomeDataset
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import roc_auc_score


def preprocess_data(df):
    df.fnlwgt = ((df.fnlwgt - df.fnlwgt.mean()) / df.fnlwgt.std())
    df['made_money'] = (df['capital.gain']> 0).astype(int)
    df['loaded'] = (df['capital.gain'] == 99_999).astype(int)
    df['full_time'] = (df['hours.per.week'] >= 40).astype(int)
    df['edu_work_race'] = (df['education.num'] * df['workclass'].astype('category').cat.codes * df['race']).astype('category').cat.codes ** (1/3)
    df['fnlwgt_squared'] = df['fnlwgt'] ** 2
    df['educated'] = (df['education.num'] > 8).astype(int)       
    df['highly_educated'] = (df['education.num'] > 13).astype(int)       
    return df



def train_loop(device, train_dataset, val_dataset, NUM_EPOCHS, PRINT_EVERY, criterion, model, loader, opt):
    runtime_stats = []
    model.train()
    best_val_auc = 0
    best_model = None
    for epoch in range(NUM_EPOCHS+1):        
        for i, sample in enumerate(loader):
            train_X, embed_X, train_y = sample['X'].to(device), sample['X_embed'].to(device), sample['y'].to(device)                    
            pred_y = model.forward(train_X, embed_X)
            loss = criterion(pred_y, train_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if epoch % PRINT_EVERY == 0:
            model.eval()

            pred_y = model(train_dataset.X, train_dataset.X_embed)                                    
            train_loss = criterion(pred_y, train_dataset.y).detach()   
            train_auc = roc_auc_score(train_dataset.y.cpu().numpy(), pred_y.detach().cpu().numpy())            

            pred_y = model(val_dataset.X, val_dataset.X_embed)                                    
            val_loss = criterion(pred_y, val_dataset.y).detach()        
            val_auc = roc_auc_score(val_dataset.y.cpu().numpy(), pred_y.detach().cpu().numpy())  

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = deepcopy(model)

            model.train()

            runtime_stats.append([epoch, train_loss, val_loss, train_auc, val_auc])
            print(f"epoch: {epoch}, width: {model.width}, depth: {model.depth}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}, train auc: {train_auc:.3f}, val auc: {val_auc:.3f}")


    runtime_df = pd.DataFrame(runtime_stats, columns=['epoch', 'train_loss', 'val_loss', 'train_auc', 'val_auc'])
    return best_model, best_val_auc, runtime_df

if __name__ == '__main__':
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    
    df = pd.read_csv('./data/train_final.csv')    
    test_df = pd.read_csv('./data/test_final.csv').drop(columns=['ID'])    

    df = preprocess_data(df)
    test_df = preprocess_data(test_df)
    
    X_embed_df = df.drop(columns=['income>50K']).select_dtypes('object').astype('category')
    X_embed_test = test_df.select_dtypes('object').astype('category')        
    
    embed_counts = {n: len(col.cat.categories) for n, col in X_embed_df.items() if len(col.cat.categories) > 2}    
    embed_cols = embed_counts.keys()
    embed_sizes = [(c, min(50, (c+1)//2)) for _, c in embed_counts.items()]    

    train_len = int(df.shape[0] * .8)
    
    train_numerical_df = df.iloc[:train_len, :].select_dtypes('number')
    val_numerical_df = df.iloc[train_len:, :].select_dtypes('number')

    test_numerical_df = test_df.select_dtypes('number').values    
    X_embed_test = X_embed_test.apply(lambda x: x.cat.codes).values

    X_train_embed_df = X_embed_df.iloc[:train_len, :]
    X_val_embed_df = X_embed_df.iloc[train_len:, :]    
    train_dataset = IncomeDataset(train_numerical_df, X_train_embed_df, device, dtype)
    val_dataset = IncomeDataset(val_numerical_df, X_val_embed_df, device, dtype)

    input_dim = train_dataset.shape[1]

    NUM_EPOCHS = 500 
    BATCH_SIZE = 8192
    PRINT_EVERY = 100
    lr = 1e-3
    
    widths = [16,32,64,128,256]
    depths = [2,4,6,8]

    all_runtimes_stats = []
    best_val = 0
    overall_best_model = None
    best_runtime_df = None
    for width in widths:
        for depth in depths:
            criterion = nn.BCELoss()
            model = BinaryClassification(input_dim, width, depth, device, embed_sizes)
            sampler = WeightedRandomSampler(train_dataset.get_balanced_weights(), num_samples=BATCH_SIZE, replacement=False)
            loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
            opt = optim.AdamW(model.parameters(), lr=lr)
            
            best_model, best_val_auc, runtime_df = train_loop(device, train_dataset, val_dataset, NUM_EPOCHS, PRINT_EVERY, criterion, model, loader, opt)

            all_runtimes_stats.append({'width': width, 'depth': depth, 'auc_series': runtime_df.val_auc})

            if best_val_auc > best_val:
                overall_best_model = deepcopy(best_model)
                best_val = best_val_auc
                best_runtime_df = runtime_df.copy()
            
            print()
    

    overall_best_model.eval()
    test_preds = overall_best_model(torch.from_numpy(test_numerical_df).to(device=device, dtype=dtype), torch.from_numpy(X_embed_test).to(device=device, dtype=torch.int64))
    test_preds = test_preds.detach().cpu().numpy().flatten()
    submission_df = pd.DataFrame(list(zip(range(1,test_preds.shape[0] + 1), test_preds)), columns=['ID', 'Prediction'])
    submission_df.to_csv('submission.csv', index=False)

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(18,14))
    plt.plot(best_runtime_df.epoch, best_runtime_df.train_loss, label='Train Loss')
    plt.plot(best_runtime_df.epoch, best_runtime_df.val_loss, label='Val Loss')    
    plt.plot(best_runtime_df.epoch, best_runtime_df.train_auc, label='Train AUC')
    plt.plot(best_runtime_df.epoch, best_runtime_df.val_auc, label='Val AUC')
    plt.legend()
    plt.savefig('Performance_graph')

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(18,14))
    for stats in all_runtimes_stats:
        plt.plot(stats['auc_series'], label=f"Val AUC width: {stats['width']}, depth: {stats['depth']}")
    plt.legend()
    plt.savefig('Experiments_graph')



    

    




    

class Dataset(Dataset):

  'Characterizes a dataset'
  def __init__(self, ID, data_label):
        'Init'

        self.data_label = data_label
        self.ID = ID

  def __len__(self):
        'number of samples'
        return len(self.ID)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        identity = self.ID[index]

        # Load data and get label
        X_input = torch.load('data/' + identity + '.pt')
        y_output = self.data_label[identity]

        return (X_input, y_output)

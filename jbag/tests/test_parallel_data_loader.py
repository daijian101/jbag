import os

from cavass.ops import read_cavass_file

from jbag.samplers import PreloadDataset


class PreloadJSONImageDataset(PreloadDataset):
    def __init__(self,
                 json_file_name_list,
                 json_dir,
                 patch_size,
                 n_patches_per_sample,
                 n_samples_alive,
                 ):
        """
        JSON format image dataset. Recently, I like JSON.
        Args:
            json_file_name_list:
            json_dir:
            label_dict:
            label_element: element in label obj that indicates gt segmentation
        """

        self.json_dir = json_dir
        super().__init__(json_file_name_list, patch_size, n_patches_per_sample, n_samples_alive)
        pass

    def get_sample_data(self, index):
        print(f'reading {index}')
        image_obj = read_cavass_file(os.path.join(self.json_dir, index, f'{index}-CT.IM0'))
        sat_data = read_cavass_file(os.path.join(self.json_dir, index, f'{index}_SAT.BIM'))
        data = {'image': image_obj, 'subject': index, 'sat': sat_data}
        return data

    def get_label_data4sampling(self, data):
        return data['sat']


def main():
    data_path = '/Users/jiandai/data/W-DS1/All_original-CT-254'
    subjects = [each for each in os.listdir(data_path)]
    dataset = PreloadJSONImageDataset(subjects, data_path, (64, 64, 32), 1, 10)
    for batch in dataset:
        print(batch['subject'])


if __name__ == '__main__':
    main()

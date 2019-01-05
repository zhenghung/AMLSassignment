import pandas as pd


class Utilities:

    @staticmethod
    def to_dataframe(name_list, pred_list):
        names = [name.zfill(10) for name in name_list]
        df = pd.DataFrame({"Filename": names,
                           "Predictions": pred_list})
        df = df.sort_values('Filename')
        return df

    @staticmethod
    def save_csv(dataframe, accuracy, filename):
        # Saving Results into csv
        csv_file = "results/{}.csv".format(filename)
        dataframe.to_csv(csv_file, index=False)

        data = open(csv_file, 'r').readlines()[1:]
        data.insert(0, '{},,\n'.format(accuracy))
        open(csv_file, 'w').writelines(data)
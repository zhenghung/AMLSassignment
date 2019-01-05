import matplotlib.pyplot as plt


class Plotting:
    @staticmethod
    def plot_history(history, directory, epochs_size,feature_tested,suffix,save=True, show=False):
        
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['acc'])
        ax1.plot(history.history['val_acc'])
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'validation'], loc='lower right')
        # plt.show(block=False)
        if save:
            plt.savefig(directory+'plots/{}_{}_accuracy_{}.png'.format(epochs_size,feature_tested,suffix))
        else:
            plt.show(block=False)

        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'validation'], loc='upper right')
        # plt.show(block=False)
        if save:
            plt.savefig(directory+'plots/{}_{}_loss_{}.png'.format(epochs_size,feature_tested,suffix))
        else:
            plt.show(block=False)

        if show:
            plt.show()

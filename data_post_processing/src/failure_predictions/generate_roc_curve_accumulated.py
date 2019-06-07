from toolkit import *
import os
if not 'DISPLAY' in os.environ.keys():
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    processed_data_folder = '/home/whitesea/workspace/data_post_processing/processed_data/'
    model_file = sys.argv[1] + 'failure_model.h5'
    splitfile = sys.argv[1] + 'training_validation_split.txt'
    datafile_pattern = 'accumulator_data_5.npy'
    TPR,FPR,predictions = generate_roc_curve_from_outputs(processed_data_folder,model_file,splitfile,datafile_pattern)
    plt.plot(FPR,TPR,'-o')
    plt.plot([0,1],[0,1],'--r')
    plt.title('ROC curve of failure_model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(sys.argv[1] + 'roc_curve.png')
    np.save(sys.argv[1] + 'predictions.npy',predictions)

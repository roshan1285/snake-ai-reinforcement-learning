import matplotlib.pyplot as plt
from IPython import display

plt.ion() # Turn on interactive mode

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf() # Clear the old plot
    
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Plot two lines
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    
    plt.ylim(ymin=0) # Start Y-axis at 0
    
    # Add text to the end of the line so we can read the exact number
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.show(block=False) # Show without stopping the script
    plt.pause(.1) # Brief pause to let the GUI update
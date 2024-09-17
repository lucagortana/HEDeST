import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_stats(predictions, ct_list, metric="predicted"):
    stats = {}

    if metric == "predicted":
        for _, prediction in predictions.items():
            predicted_class = prediction['predicted_class']
            max_prob = max(prediction['probabilities'])

            if predicted_class not in stats:
                stats[predicted_class] = {
                    'probs': [],
                    'count': 0
                }

            stats[predicted_class]['probs'].append(max_prob)
            stats[predicted_class]['count'] += 1

        data = []
        for class_id, class_stats in stats.items():
            class_probs = class_stats['probs']
            row = [
                class_id,
                ct_list[class_id],
                np.min(class_probs),
                np.max(class_probs),
                np.median(class_probs),
                np.mean(class_probs),
                class_stats['count']
            ]
            data.append(row)

        columns = ['Class', 'CT', 'Min Prob', 'Max Prob', 'Median Prob', 'Mean Prob', 'Cell Count']
        df_stats = pd.DataFrame(data, columns=columns)

    elif metric == "all":
        num_classes = len(predictions[next(iter(predictions))]['probabilities'])

        for class_id in range(num_classes):
            stats[class_id] = {
                'probs': [],
                'count': 0
            }

        for _, prediction in predictions.items():
            prob_vector = prediction['probabilities']
            for class_id, prob in enumerate(prob_vector):
                stats[class_id]['probs'].append(prob)
                stats[class_id]['count'] += 1

        data = []
        for class_id, class_stats in stats.items():
            class_probs = class_stats['probs']
            row = [
                class_id,
                ct_list[class_id],
                np.min(class_probs),
                np.max(class_probs),
                np.median(class_probs),
                np.mean(class_probs)
            ]
            data.append(row)

        columns = ['Class', 'CT', 'Min Prob', 'Max Prob', 'Median Prob', 'Mean Prob']
        df_stats = pd.DataFrame(data, columns=columns)

    else:
        raise ValueError("Invalid metric. Choose 'predicted' or 'all'.")
    
    df_stats = df_stats.sort_values(by='Class', ascending=True).reset_index(drop=True)
    df_stats = df_stats.set_index('Class')

    return df_stats

def generate_dicts_viz_pred(nuc_dict, class_dict, ct_list):
    """
    Updates the nuclear dictionary with the predicted classes.
    """
    for i, (key, _) in enumerate(nuc_dict['nuc'].items()):
        if str(i) in class_dict:
            nuc_dict['nuc'][key]['type'] = class_dict[str(i)]['predicted_class']
    
    return nuc_dict, generate_color_dict(ct_list)

def generate_color_dict(list, palette='hsv'):
    """
    Generate a dictionary of colors for each class in the list.
    """
    color_dict = {}
    num_classes = len(list)
    
    cmap = plt.get_cmap(palette, num_classes)
    
    for i, class_name in enumerate(list):
        color = cmap(i)[:3]
        color = [int(255 * c) for c in color]
        color_dict[str(i)] = [class_name, color]
    
    return color_dict

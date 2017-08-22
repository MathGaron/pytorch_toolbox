from pytorch_toolbox.visualization.visdom_handler import VisdomHandler


def console_print(loss, data_time, batch_time, scores, istrain):
    state = "Train" if istrain else "Valid"
    print(' {}\t || Loss: {:.3f} | Load Time {:.3f}s | Batch Time {:.3f}s'.format(state, loss, data_time, batch_time))
    for i, acc in enumerate(scores):
        print('\t || Acc {}: {:.3f}'.format(i, acc))


def visdom_print(loss, data_time, batch_time, scores, istrain):
    state = "Train" if istrain else "Valid"
    vis = VisdomHandler()
    vis.visualize(loss, '{} loss'.format(state))
    vis.visualize(data_time, '{} data load time'.format(state))
    vis.visualize(batch_time, '{} batch processing time'.format(state))
    for i, acc in enumerate(scores):
        vis.visualize(acc, '{} score {}'.format(i, state))



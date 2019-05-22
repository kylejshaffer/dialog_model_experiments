import argparse, os
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib

def freeze_graph(model_path, output_node_names, output_path):
    model_base_path = os.path.split(model_path)[0]
    if not os.path.exists(model_base_path):
        raise AssertionError("directory doesn't exist")

    if not output_node_names:
        print("Please supply name of a node to output to --output_node_names")
        return -1

    with tf.Session(graph=tf.Graph()) as sess:
        # Load in model graph structure
        saver = tf.train.import_meta_graph(model_path + '.meta')
        # Load in weights under session
        saver.restore(sess, model_path)

        # Looking at TF operation
        for n in tf.get_default_graph().as_graph_def().node:
            # if ('logits' in n.name) or (n.name == 'x_input'):
            if ('gradients' not in n.name):
                print(n.name)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=tf.get_default_graph().as_graph_def(), output_node_names=output_node_names)

        # Serialize and dump
        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('{} ops in final graph'.format(len(output_graph_def.node)))
    return output_graph_def

def load_graph(graph_path, name):
    with tf.gfile.GFile(graph_path, 'rb') as infile:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(infile.read())
    return graph_def

def create_inference_graph(input_graph_def, input_nodes, output_nodes, output_graph_name):
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, input_nodes,
                                                output_nodes, tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile(output_graph_name, 'w') as outgraph:
        outgraph.write(output_graph_def.SerializeToString())

# run it
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    root_path = '/data/users/kyle.shaffer/chat_models'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(root_path, 'lstm_chatbot_epoch36.ckpt'))
    parser.add_argument('--output_node_names', type=str, default='x, log_softmax')
    parser.add_argument('--output_path', type=str, default=os.path.join(root_path, 'lstm_chatbot_epoch36.pb'))

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # output_node_names = [n.strip() for n in args.output_node_names.split(',')]
    output_node_names = ['x_input',  'decoder_input', 'logits/MatMul']
    freeze_graph(model_path=args.model_path, output_node_names=output_node_names,
                output_path=args.output_path)
    print('frozen graph created...')
    input_graph_def = load_graph(graph_path=args.output_path, name='lstm_chatbot')
    print('frozen graph read in...')
    # create_inference_graph(input_graph_def=input_graph_def, input_nodes=[output_node_names[0]],
    #                        output_nodes=[output_node_names[-1]], output_graph_name='models/bidi_lm_opt.pb')
    # print('optimized graph created.')
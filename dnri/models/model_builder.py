from . import encoders
from . import decoders
from . import nri
from . import dnri
from . import dnri_dynamicvars
from . import recurrent_baselines
from . import recurrent_baselines_dynamicvars
import os


def build_model(params, verbose=False):
    if params['model_type'] == 'dnri':
        dynamic_vars = params.get('dynamic_vars', False)
        if dynamic_vars:
            model = dnri_dynamicvars.DNRI_DynamicVars(params)
        else:
            model = dnri.DNRI(params)
        model_type = "dNRI MODEL"

    elif params['model_type'] == 'fc_baseline':
        dynamic_vars = params.get('dynamic_vars', False)
        if dynamic_vars:
            model = recurrent_baselines_dynamicvars.FullyConnectedBaseline_DynamicVars(params)
        else:
            model = recurrent_baselines.FullyConnectedBaseline(params)
        model_type = "FCBaseline"
    else:
        num_vars = params['num_vars']
        graph_type = params['graph_type']

        # Build Encoder
        encoder = encoders.RefMLPEncoder(params)
        model_type = "ENCODER"

        # Build Decoder
        decoder = decoders.GraphRNNDecoder(params)
        model_type = "DECODER"
        if graph_type == 'dynamic':
            model = nri.DynamicNRI(num_vars, encoder, decoder, params)
        else:
            model = nri.StaticNRI(num_vars, encoder, decoder, params)

    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    if params['gpu']:
        model.cuda()
    
    if verbose:
        print(model_type+": ", model)
    return model


from gluonts.mx.distribution.lowrank_gp import LowrankGPOutput
from typing import List

from gluonts.mx.distribution import bijection
#from gluonts.mx.distribution.multivariate_independent_gaussian import (
#    MultivariateIndependentGaussianOutput,
#)
from gluonts.mx.distribution.transformed_distribution import (
    TransformedDistribution
)
from gluonts.mx.distribution import (
    LowrankMultivariateGaussianOutput,
    MultivariateGaussianOutput,
    NegativeBinomialOutput
)

#from model.deepnegpol.negpol_distr import NegativeBinomial_DirichletMultinomialOutput

from model.deepvar import DeepVAREstimator
from model.gpvar import GPVAREstimator
from model.deepar import DeepAREstimator
from model.deepnegpol import DeepNEGPOLEstimator
from model.lstmmaf import TempFlowEstimator
from model.deepvar_torch import DeepVAREstimator as DeepVAREstimator_torch


from hyperparams import Hyperparams








def trainer_from_params(
    params: Hyperparams,
    target_dim: int,
    mx_type: bool = True,
    low_rank: bool = True,
    hybridize: bool = None,
):
    
    # find a batch_size so that 1024 examples are used for SGD and cap the value in [8, 32]
    batch_size = params.batch_size
    if target_dim > 1000 :#or not low_rank:
        # avoid OOM
        batch_size = 4
        
    
    if mx_type:
       from gluonts.mx.trainer import Trainer
       return Trainer(
        epochs=params.epochs,
        batch_size=batch_size,  # todo make it dependent from dimension
        learning_rate=params.learning_rate
        if low_rank
        else params.learning_rate_fullrank,
        minimum_learning_rate=params.minimum_learning_rate,
        patience=params.patience,
        num_batches_per_epoch=params.num_batches_per_epoch,
        hybridize=hybridize if hybridize is not None else params.hybridize,
        ctx='cpu'
        )

       
    else:
       from pts import Trainer
       import os
       os.environ['CUDA_VISIBLE_DEVICES'] = "0"
       import torch
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       #device = torch.device("cpu")


       return Trainer(
        epochs=params.epochs,
        batch_size=batch_size,  # todo make it dependent from dimension
        learning_rate=params.learning_rate
        if low_rank
        else params.learning_rate_fullrank,
        minimum_learning_rate=params.minimum_learning_rate,
        patience=params.patience,
        num_batches_per_epoch=params.num_batches_per_epoch,
        hybridize=hybridize if hybridize is not None else params.hybridize,
        device = device
        )

    # batch_size = 512 // target_dim
    # batch_size = min(max(8, 1024 // max(batch_size, 1)), 32)
    # if not low_rank:
    #    # avoid OOM
    #    batch_size = 4






class LowrankMultivariateGaussianOutputTransformed(
    LowrankMultivariateGaussianOutput
):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None):
        base_dist = super(
            LowrankMultivariateGaussianOutputTransformed, self
        ).distribution(dist_args, scale)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, self.transform)
        else:
            return TransformedDistribution(base_dist, self.transform)




class GPLowrankMultivariateGaussianOutputTransformed(LowrankGPOutput):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None, dim=None):
        base_dist = super(
            GPLowrankMultivariateGaussianOutputTransformed, self
        ).distribution(dim = dim, scale = scale, distr_args = dist_args)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, self.transform)
        else:
            return TransformedDistribution(base_dist, self.transform)


class MultivariateGaussianOutputTransformed(MultivariateGaussianOutput):
    def __init__(self, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def distribution(self, dist_args, scale=None):
        base_dist = super(
            MultivariateGaussianOutputTransformed, self
        ).distribution(dist_args, scale)
        if isinstance(self.transform, List):
            return TransformedDistribution(base_dist, self.transform)

        else:
            return TransformedDistribution(base_dist, self.transform)


#class MultivariateIndependentGaussianOutputTransformed(
#    MultivariateIndependentGaussianOutput
#):
#    def __init__(self, transform, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.transform = transform
#
#    def distribution(self, dist_args, scale=None):
#        base_dist = super(
#            MultivariateIndependentGaussianOutputTransformed, self
#        ).distribution(dist_args, scale)
#        if isinstance(self.transform, List):
#            return TransformedDistribution(base_dist, self.transform)
#        else:
#            return TransformedDistribution(base_dist, self.transform)


def distr_output_from_params(
    target_dim, diagonal_only, transform, low_rank, params
):
    if not low_rank:
        likelihood = MultivariateGaussianOutputTransformed(
            transform, dim=target_dim
        )
    else:
        if diagonal_only:
            likelihood = MultivariateIndependentGaussianOutputTransformed(
                transform, dim=target_dim
            )
        else:
            likelihood = LowrankMultivariateGaussianOutputTransformed(
                transform,
                dim=target_dim,
                rank=min(params.rank, target_dim) if not diagonal_only else 1,
            )
    return likelihood


def make_multivariate_estimator(
    low_rank: bool,
    diagonal_only: bool,
    cdf: bool = True,
    rnn: bool = True,
    scaling: bool = False,
):
    def make_model(
        freq: str, prediction_length: int, target_dim: int, params: Hyperparams
    ):
        transform = []

        distr_output = distr_output_from_params(
            target_dim=target_dim,
            diagonal_only=diagonal_only,
            transform=transform,
            low_rank=low_rank,
            params=params,
        )

        context_length = 3*prediction_length
        

        estimator = DeepVAREstimator(
            target_dim=target_dim,
            num_cells=params.num_cells,
            num_layers=params.num_layers,
            dropout_rate=params.dropout_rate,
            prediction_length=prediction_length,
            context_length=context_length,
            cell_type=params.cell_type if rnn else "time-distributed",
            freq=freq,
            pick_incomplete=False,
            distr_output=distr_output,
            conditioning_length=params.conditioning_length,
            trainer=trainer_from_params(
                params=params,
                target_dim=target_dim,
                low_rank=low_rank,
                hybridize=params.hybridize,
            ),
            scaling=scaling,
            use_marginal_transformation=cdf,
            lags_seq=params.lags_seq,
        )
        return estimator

    return make_model






def make_multivariate_ind_estimator(
    scaling: bool = False,
):
    def make_model(
        freq: str, prediction_length: int, target_dim: int, params: Hyperparams
    ):
        from pts.modules import NormalOutput

        distr_output = NormalOutput(target_dim)

        context_length = 3*prediction_length
        

        estimator = DeepVAREstimator_torch(
            input_size = 298,
            freq = freq,
            prediction_length = prediction_length,
            target_dim = target_dim,
            trainer=trainer_from_params(
                        params=params,
                        mx_type = False,
                        target_dim=target_dim,
                        hybridize=params.hybridize,
                ),
            context_length=context_length,
            num_layers=params.num_layers,
            num_cells=params.num_cells,
            cell_type = "LSTM",
            num_parallel_samples= params.num_eval_samples,
            dropout_rate=params.dropout_rate,
            use_feat_dynamic_real=True,
            use_feat_static_cat = False,
            use_feat_static_real = False,
            distr_output=distr_output,
            scaling = scaling,
            pick_incomplete = False,
            lags_seq=params.lags_seq,
        )
        return estimator

    return make_model




def make_gp_estimator(
    cdf: bool = True, rnn: bool = True, scaling: bool = False
):
    def _make_gp_estimator(
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Hyperparams,# = Hyperparams(),
    ):

        context_length = 3*prediction_length


        transform = []

        distr_output = GPLowrankMultivariateGaussianOutputTransformed(
            transform,
            dim=target_dim,
            rank=min(params.rank, target_dim),
            dropout_rate=params.dropout_rate,
        )

        return GPVAREstimator(
            target_dim=target_dim,
            num_cells=params.num_cells,
            num_layers=params.num_layers,
            dropout_rate=params.dropout_rate,
            prediction_length=prediction_length,
            context_length=context_length,
            cell_type=params.cell_type if rnn else "time-distributed",
            target_dim_sample=params.target_dim_sample,
            lags_seq=params.lags_seq,
            pick_incomplete=False,
            conditioning_length=params.conditioning_length,
            scaling=scaling,
            freq=freq,
            rank = params.rank,
            use_marginal_transformation=cdf,
            distr_output=distr_output,
            trainer=trainer_from_params(
                params=params,
                target_dim=target_dim,
                hybridize=params.hybridize,
            ),
        )

    return _make_gp_estimator







def make_deepar_estimator(
    scaling: bool = False
):
    def _make_deepar_estimator(
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Hyperparams ,#= Hyperparams(),
    ):

        context_length = 3*prediction_length

        distr_output = NegativeBinomialOutput()

        return DeepAREstimator(
                    freq=freq,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    num_cells=params.num_cells,
                    num_layers=params.num_layers,
                    cell_type=params.cell_type,
                    dropout_rate=params.dropout_rate,
                    trainer=trainer_from_params(
                        params=params,
                        target_dim=1,
                        hybridize=params.hybridize,
                    ),
                    cardinality=[target_dim],
                    distr_output = distr_output,
                    scaling = scaling,
                    lags_seq = params.lags_seq,
                    use_feat_dynamic_real=True,
                    use_feat_static_cat=True,
                   )
    
    


    return _make_deepar_estimator






def make_deepnegpol_estimator(
    scaling: bool = False
):
    def _make_deepnegpol_estimator(
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Hyperparams ,#= Hyperparams(),
    ):

        context_length = 3*prediction_length


        return DeepNEGPOLEstimator(
                input_size1 = 103,
                input_size2 = 271,
                num_cells1 = params.num_cells1,
                num_cells2 = params.num_cells2,
                num_layers1=params.num_layers,
                num_layers2=params.num_layers,
                dropout_rate=params.dropout_rate,
                pick_incomplete=False,
                target_dim=target_dim,
                prediction_length=48,#prediction_length,
                context_length = 3*48,#context_length,
                freq=freq,
                scaling=scaling,
                use_feat_dynamic_real = True,
                lags_seq = params.lags_seq,
                trainer=trainer_from_params(
                        params=params,
                        mx_type = False,
                        target_dim=target_dim,
                        hybridize=params.hybridize,
                ),
            )
    
        

    
    


    return _make_deepnegpol_estimator





def make_lstmMaf_estimator(
    scaling: bool = False
):
    def _make_lstmMaf_estimator(
        freq: str,
        prediction_length: int,
        target_dim: int,
        params: Hyperparams ,#= Hyperparams(),
    ):

        context_length = 3*prediction_length


  
        return TempFlowEstimator(
		    target_dim=target_dim,
		    prediction_length=48,
		    context_length = 3*48,
		    cell_type='LSTM',
		    input_size=296,
		    num_cells = params.num_cells,
		    num_layers=params.num_layers,
		    dropout_rate=params.dropout_rate,
		    freq=freq,
		    scaling=scaling,
		    dequantize=True,
		    lags_seq = params.lags_seq,
		    flow_type='MAF',
		    trainer=trainer_from_params(
                        params=params,
                        mx_type = False,
                        target_dim=target_dim,
                        hybridize=params.hybridize,
                )
		)
    



    return _make_lstmMaf_estimator




models_dict = {
    "DeepNegPol": make_deepnegpol_estimator(
        scaling=True
    ),
    "DeepAR": make_deepar_estimator(
        scaling=True
    ),
    "LSTMIndScaling": make_multivariate_ind_estimator(
        scaling=True
    ),
    "LSTMInd": make_multivariate_ind_estimator(
    ),
    "LSTMFRScaling": make_multivariate_estimator(
        low_rank=False, diagonal_only=False, cdf=False, scaling=True
    ),
    "LSTMFR": make_multivariate_estimator(
        low_rank=False, diagonal_only=False, cdf=False
    ),
    "LSTMCOP": make_multivariate_estimator(
        low_rank=True, diagonal_only=False, cdf=True
    ),
    "GPCOP": make_gp_estimator(cdf=True, rnn=True),
    "GP": make_gp_estimator(cdf=False, rnn=True),
    "GPScaling": make_gp_estimator(cdf=False, rnn=True, scaling=True),
    "LSTMMAF" : make_lstmMaf_estimator(scaling=True) 
}


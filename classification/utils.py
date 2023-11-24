import torch
import logging
import numpy as np
import torch.distributed as dist

from datasets.imagenet_subsets import IMAGENET_D_MAPPING
from torch.nn.parallel import DistributedDataParallel
import os
logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict, data, predictions):
    """
    Create a dictionary which separates the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domains and the content is [labels, predictions]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: predictions of the model
    :return: updated result dict
    """

    imgs = data[0][0] if isinstance(data[0], list) else data[0]

    for i in range(imgs.shape[0]):
        label, domain = data[1][i], data[2][i]
        if domain in domain_dict.keys():
            domain_dict[domain].append([label.item(), predictions[i].item()])
        else:
            domain_dict[domain] = [[label.item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    dom_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        logger.info(f"{key:<20} error: {error:.2%}")
    avg_err = 1 - sum(correct) / sum(num_samples)
    logger.info(f"Average error: {avg_err:.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            correct += (predictions == labels.to(device)).float().sum()

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class CustomDistributedDataParallel(DistributedDataParallel):
    """A wrapper class over DDP that relay "module" attribute."""

    def __init__(self, model, **kwargs):
        super(CustomDistributedDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        try:
            return super(CustomDistributedDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

NUM_CLASSES = {"domainnet-126": 126, "VISDA-C": 12}


def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )

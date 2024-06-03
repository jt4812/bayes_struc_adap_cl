from src.data.mix_cifar import get_split_cifarmix_dataloaders
from src.data.mnist_cl import get_not_mnist_dataloaders, get_perm_mnist_dataloaders, \
    get_split_dataloaders
from src.data.tiny_imagenet import get_split_tiny_imagenet_dataloaders


def get_cl_dataloaders(exp_dict, batch_size):
    use_distillation = ("use_distillation" in exp_dict and exp_dict["use_distillation"])
    assert exp_dict["dataset"] in [
        "not_mnist", "one_mnist",  "one_fashion_mnist",
        "perm_mnist", "split_mnist", "split_fashion_mnist",
        "split_cifar10", "split_cifar100_5", "split_cifar100_10", "split_cifar100_20", "split_cifarmix",
        "split_tiny_imagenet_10",
    ]

    if "test_batch_size" in exp_dict:
        test_batch_size = exp_dict["test_batch_size"]
    else:
        test_batch_size = None

    use_validation_dataset = ("use_validation_dataset" in exp_dict) and exp_dict["use_validation_dataset"]

    if "split_tiny_imagenet" in exp_dict["dataset"]:
        return get_split_tiny_imagenet_dataloaders(exp_dict["batch_size"], exp_dict["n_tasks"])
    elif "split_cifarmix" in exp_dict["dataset"]:
        return get_split_cifarmix_dataloaders(batch_size, use_validation_dataset=use_validation_dataset)
    elif exp_dict["dataset"] == "one_mnist":
        return get_split_dataloaders(
            batch_size, True, "task_il", dataset=exp_dict["dataset"], input_as_image="input_as_image" in exp_dict,
            test_batch_size=test_batch_size, use_distillation=use_distillation)
    elif exp_dict["dataset"] == "one_fashion_mnist":
        return get_split_dataloaders(
            batch_size, True, "task_il", dataset=exp_dict["dataset"], input_as_image="input_as_image" in exp_dict,
            test_batch_size=test_batch_size, use_distillation=use_distillation)
    elif exp_dict["dataset"] == "not_mnist":
        return get_not_mnist_dataloaders(
            batch_size, True, "task_il", dataset=exp_dict["dataset"], input_as_image="input_as_image" in exp_dict,
            use_distillation=use_distillation)
    elif exp_dict["dataset"] == "perm_mnist":
        return get_perm_mnist_dataloaders(
            batch_size, cl_type=exp_dict["cl_type"], n_tasks=exp_dict["n_tasks"], use_distillation=use_distillation,
            use_validation_dataset=use_validation_dataset)
    elif "split" in exp_dict["dataset"]:
        return get_split_dataloaders(
            batch_size, exp_dict["single_head"], exp_dict["cl_type"], dataset=exp_dict["dataset"],
            input_as_image="input_as_image" in exp_dict, test_batch_size=test_batch_size,
            use_distillation=use_distillation, use_validation_dataset=use_validation_dataset)

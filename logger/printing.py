def print_loss_and_estimates(loss_dict, estimates, epoch):
    if epoch % 10 == 0:
        print("--------------------------------")
        print(f"Iteration {epoch}")
        for model_name, loss in loss_dict.items():
            if model_name == "total":
                print(f"Total Loss: {loss.item():.4f}")
                continue
            elif model_name == "SecondMoment":
                print(f"{model_name}: Loss: {loss.item():.4f}")
                continue
            else:
                print(f"{model_name}: Estimated: {estimates[model_name].mean().item():.4f}, Loss: {loss.item():.4f}")

import matplotlib.pyplot as plt

class DiffusionTensorBoardLogger:
    """
    Logger TensorBoard pour modèles de diffusion sur systèmes dynamiques
    + UQ + OOD + trajectoires
    """

    def __init__(self, log_dir="runs/exp"):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_loss(self, loss, name="train/loss"):
        self.writer.add_scalar(name, loss, self.step)

    def log_uq(self, uq_score, name="uq/total"):
        self.writer.add_scalar(name, uq_score, self.step)

    def log_ood(self, ood_score, name="ood/score"):
        self.writer.add_scalar(name, ood_score, self.step)

    def log_diversity(self, diversity, name="diversity"):
        self.writer.add_scalar(name, diversity, self.step)

    def log_learning_rate(self, lr):
        self.writer.add_scalar("train/lr", lr, self.step)

    def log_trajectory_sample(self, trajs, name="samples/trajectories"):
        """
        trajs: (N, T)
        On log une grille de trajectoires
        """
        fig = self._plot_trajectories(trajs)
        self.writer.add_figure(name, fig, self.step)

    def log_score_histogram(self, scores, name="ood/hist"):
        self.writer.add_histogram(name, scores, self.step)

    def step_forward(self):
        self.step += 1

    def close(self):
        self.writer.close()

    def _plot_trajectories(self, trajs):
        fig, ax = plt.subplots()

        for i in range(min(len(trajs), 10)):
            ax.plot(trajs[i].cpu().numpy(), alpha=0.7)

        ax.set_title("Generated trajectories")
        ax.set_xlabel("time")
        ax.set_ylabel("state")

        return fig
    

"""
for epoch in range(num_epochs):

    for x0, c in dataloader:

        t = torch.randint(1, T, (x0.shape[0],))

        eps = torch.randn_like(x0)

        alpha_bar_t = alpha_bar[t].unsqueeze(-1)

        x_t = (
            torch.sqrt(alpha_bar_t) * x0 +
            torch.sqrt(1 - alpha_bar_t) * eps
        )

        eps_pred = model(x_t, t, c)

        loss = ((eps - eps_pred)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------
        # METRICS
        # -------------------------

        # UQ (ex: erreur de débruitage)
        uq = loss.detach()

        # OOD proxy (option simple)
        ood = eps_pred.std()

        # Diversité (si tu génères des samples)
        trajs = sample_trajectories(model, c, num_samples=10)
        diversity = trajs.var(dim=0).mean()

        # -------------------------
        # LOGGING
        # -------------------------

        logger.log_loss(loss.item())
        logger.log_uq(uq.item())
        logger.log_ood(ood.item())
        logger.log_diversity(diversity.item())

        logger.log_trajectory_sample(trajs)

        logger.step_forward() 
"""


EVOLUTION : Dashboard avec streamlit
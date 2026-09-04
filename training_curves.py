class TrainingCurvePlotter:
    """绘制训练集和验证集曲线的共用模块。

    调用方只需要在每轮训练结束后调用 update，训练完成后调用 show。
    enabled=False 时所有方法都是空操作，也不会导入 Matplotlib。
    """

    def __init__(self, enabled: bool, title: str):
        self.enabled = enabled
        if not enabled:
            return

        # 仅在用户启用曲线时导入，普通训练不承担绘图库启动开销。
        import matplotlib.pyplot as plt

        self._plt = plt
        self._epochs: list[int] = []
        self._train_losses: list[float] = []
        self._validation_losses: list[float] = []
        self._train_accuracies: list[float] = []
        self._validation_accuracies: list[float] = []
        self._train_character_accuracies: list[float] = []
        self._validation_character_accuracies: list[float] = []

        plt.ion()
        (
            self._figure,
            (self._loss_axis, self._accuracy_axis, self._character_accuracy_axis),
        ) = plt.subplots(
            1,
            3,
            figsize=(17, 4),
        )
        self._figure.suptitle(title)

        self._loss_axis.set_title("Loss")
        self._loss_axis.set_xlabel("Epoch")
        self._loss_axis.set_ylabel("Loss")
        (self._train_loss_line,) = self._loss_axis.plot([], [], label="train")
        (self._validation_loss_line,) = self._loss_axis.plot([], [], label="validation")
        self._loss_axis.legend()

        self._accuracy_axis.set_title("Exact-match Accuracy")
        self._accuracy_axis.set_xlabel("Epoch")
        self._accuracy_axis.set_ylabel("Accuracy")
        self._accuracy_axis.set_ylim(0, 1)
        (self._train_accuracy_line,) = self._accuracy_axis.plot([], [], label="train")
        (self._validation_accuracy_line,) = self._accuracy_axis.plot(
            [], [], label="validation"
        )
        self._accuracy_axis.legend()

        self._character_accuracy_axis.set_title("Character Accuracy")
        self._character_accuracy_axis.set_xlabel("Epoch")
        self._character_accuracy_axis.set_ylabel("Accuracy")
        self._character_accuracy_axis.set_ylim(0, 1)
        (self._train_character_accuracy_line,) = self._character_accuracy_axis.plot(
            [], [], label="train"
        )
        (self._validation_character_accuracy_line,) = (
            self._character_accuracy_axis.plot([], [], label="validation")
        )
        self._character_accuracy_axis.legend()
        plt.show(block=False)

    def update(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        train_accuracy: float,
        validation_accuracy: float,
        train_character_accuracy: float,
        validation_character_accuracy: float,
    ) -> None:
        """记录一轮指标并刷新曲线窗口。"""
        if not self.enabled:
            return

        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._validation_losses.append(validation_loss)
        self._train_accuracies.append(train_accuracy)
        self._validation_accuracies.append(validation_accuracy)
        self._train_character_accuracies.append(train_character_accuracy)
        self._validation_character_accuracies.append(validation_character_accuracy)

        self._train_loss_line.set_data(self._epochs, self._train_losses)
        self._validation_loss_line.set_data(self._epochs, self._validation_losses)
        self._train_accuracy_line.set_data(self._epochs, self._train_accuracies)
        self._validation_accuracy_line.set_data(
            self._epochs,
            self._validation_accuracies,
        )
        self._train_character_accuracy_line.set_data(
            self._epochs,
            self._train_character_accuracies,
        )
        self._validation_character_accuracy_line.set_data(
            self._epochs,
            self._validation_character_accuracies,
        )

        self._loss_axis.relim()
        self._loss_axis.autoscale_view()
        self._accuracy_axis.relim()
        self._accuracy_axis.set_ylim(0, 1)
        self._accuracy_axis.autoscale_view(scaley=False)
        self._character_accuracy_axis.relim()
        self._character_accuracy_axis.set_ylim(0, 1)
        self._character_accuracy_axis.autoscale_view(scaley=False)
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)

    def show(self) -> None:
        """训练完成后保持曲线窗口，直到用户将它关闭。"""
        if not self.enabled:
            return
        self._plt.ioff()
        self._plt.show()

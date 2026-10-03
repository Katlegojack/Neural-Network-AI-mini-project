# Continual Learning MNIST Prototype

This Django prototype demonstrates the three continual-learning models used in the NCOS734 project:

- Naive Sequential
- Elastic Weight Consolidation (EWC)
- Experience Replay

The app accepts an uploaded handwritten digit image or a real MNIST test sample. The same input is passed to the selected model(s), and the frontend displays:

- predicted digit,
- confidence score,
- correct/incorrect result when the true digit is known,
- top three predictions.

## Run the prototype

From the repository root:

```bash
pip install -r requirements.txt
cd prototype
python manage.py prepare_models
python manage.py runserver
```

Then open:

```
http://127.0.0.1:8000/
```

### Why `prepare_models` is needed

The Git repository stores the code and MNIST data, but the trained PyTorch checkpoints are generated locally. The one-time command trains the same three models used by the project and saves them under:

```
models/
├── naive_sequential_mnist.pt
├── ewc_mnist.pt
└── experience_replay_mnist.pt
```

After that, normal prototype use only performs inference; it does not retrain the models.

## Recommended demo

For a reliable presentation, click the built-in MNIST sample digits first. Then upload your own handwritten image to demonstrate how the system handles a new image.

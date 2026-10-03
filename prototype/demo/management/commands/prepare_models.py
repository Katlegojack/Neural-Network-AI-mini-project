from django.conf import settings
from django.core.management.base import BaseCommand

from demo.training import prepare_models


class Command(BaseCommand):
    help = "Train and save Naive, EWC and Experience Replay MNIST models."

    def handle(self, *args, **options):
        output_dir = settings.REPO_ROOT / "models"
        prepare_models(settings.REPO_ROOT, output_dir)
        self.stdout.write(self.style.SUCCESS("Model preparation complete."))

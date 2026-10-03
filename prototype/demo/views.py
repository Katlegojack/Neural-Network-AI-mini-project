import csv

from django.conf import settings
from django.shortcuts import render

from .model_service import (
    MODEL_LABELS,
    get_mnist_sample,
    image_to_data_url,
    missing_models,
    predict,
    preprocess_uploaded_image,
)


def load_project_results():
    path = settings.REPO_ROOT / "results" / "final_comparison.csv"
    if not path.exists():
        return []

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "method": row["Method"],
                    "accuracy": float(row["Final_Average_Accuracy"]) * 100,
                    "forgetting": float(row["Average_Forgetting"]) * 100,
                }
            )
    return rows


def index(request):
    context = {
        "model_labels": MODEL_LABELS,
        "project_results": load_project_results(),
        "missing_models": missing_models(),
    }

    if request.method != "POST":
        return render(request, "demo/index.html", context)

    try:
        source = request.POST.get("source", "upload")
        true_digit = None

        if source == "sample":
            true_digit = int(request.POST["sample_digit"])
            tensor, image = get_mnist_sample(true_digit)
        else:
            uploaded = request.FILES.get("digit_image")
            if uploaded is None:
                raise ValueError("Please choose a digit image first.")
            tensor, image = preprocess_uploaded_image(uploaded)

            entered_digit = request.POST.get("true_digit", "").strip()
            if entered_digit:
                true_digit = int(entered_digit)
                if true_digit < 0 or true_digit > 9:
                    raise ValueError("The true digit must be between 0 and 9.")

        selected_model = request.POST.get("model", "all")
        keys = list(MODEL_LABELS.keys()) if selected_model == "all" else [selected_model]

        predictions = []
        for key in keys:
            result = predict(key, tensor)
            if true_digit is not None:
                result["correct"] = result["prediction"] == true_digit
            predictions.append(result)

        context.update(
            {
                "predictions": predictions,
                "true_digit": true_digit,
                "input_image": image_to_data_url(image),
                "selected_model": selected_model,
            }
        )

    except Exception as exc:
        context["error"] = str(exc)

    return render(request, "demo/index.html", context)

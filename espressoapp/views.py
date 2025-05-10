from django.shortcuts import render, redirect
from .forms import PredictionForm
from .utils import predict_with_knn, predict_with_decision_tree, get_knn_accuracy, get_decision_tree_accuracy
from .models import Prediction

def home(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            prediction = form.save(commit=False)
            cleaned_data = form.cleaned_data
            prediction.knn_prediction = predict_with_knn(cleaned_data)
            prediction.decision_tree_prediction = predict_with_decision_tree(cleaned_data)
            prediction.save()
            return redirect('results', pk=prediction.pk)
    else:
        form = PredictionForm()
    return render(request, 'espressoapp/home.html', {'form': form})

def results(request, pk):
    prediction = Prediction.objects.get(pk=pk)
    knn_accuracy = get_knn_accuracy()
    decision_tree_accuracy = get_decision_tree_accuracy()
    context = {
        'prediction': prediction,
        'knn_accuracy': knn_accuracy,
        'decision_tree_accuracy': decision_tree_accuracy,
    }
    return render(request, 'espressoapp/results.html', context)

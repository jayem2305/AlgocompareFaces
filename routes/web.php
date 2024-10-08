<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\FaceRecognitionController;

Route::get('/', function () {
    return view('welcome');
});
Route::get('/', [FaceRecognitionController::class, 'index'])->name('face.index');
Route::post('/upload', [FaceRecognitionController::class, 'upload'])->name('face.upload');
Route::post('/scan', [FaceRecognitionController::class, 'scan'])->name('face.scan');
Route::post('/face/compare', [FaceRecognitionController::class, 'compareFaces'])->name('face.compare');

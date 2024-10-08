<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\File;
use Symfony\Component\HttpFoundation\Response;

class FaceRecognitionController extends Controller
{
    public function scan(Request $request)
    {
        // Validate the incoming request
        $request->validate([
            'algorithm' => 'required|string|in:cnn,fastr,yolo',
            'scan_direction' => 'required|string',
            'captured_image' => 'nullable|image|mimes:jpeg,png,jpg|max:2048',
            'image' => 'nullable|image|mimes:jpeg,png,jpg|max:2048',
        ]);

        $algorithm = $request->input('algorithm');
        $scanDirection = $request->input('scan_direction');
        $capturedImage = $request->file('captured_image');
        $uploadedImage = $request->file('image');

        Log::info('Face scan initiated', [
            'algorithm' => $algorithm,
            'scan_direction' => $scanDirection,
            'user_id' => $request->user() ? $request->user()->id : 'guest'
        ]);

        // Handle image upload if provided
        if ($uploadedImage) {
            $uploadedImagePath = $this->handleImageUpload($uploadedImage, 'ID');
        } else {
            $uploadedImagePath = null;
        }

        if ($capturedImage) {
            $capturedImagePath = $this->handleImageUpload($capturedImage, $scanDirection);
        } else {
            Log::error('No captured image provided');
            return response()->json(['message' => 'No image provided'], Response::HTTP_BAD_REQUEST);
        }

        // Process the images with the chosen algorithm
        $result = $this->processAlgorithm($algorithm, $uploadedImagePath, $capturedImagePath);

        return response()->json(['message' => 'Scan completed', 'result' => $result]);
    }

    private function handleImageUpload($image, $folder)
    {
        $destinationPath = public_path('images/' . $folder);
        if (!File::exists($destinationPath)) {
            File::makeDirectory($destinationPath, 0755, true); // Ensure folder exists
        }

        $imageName = time() . '_' . $image->getClientOriginalName();
        $image->move($destinationPath, $imageName);

        return 'images/' . $folder . '/' . $imageName;
    }

    private function processAlgorithm($algorithm, $uploadedImagePath, $capturedImagePath)
    {
        if ($algorithm == "cnn") {
            Log::info('CNN algorithm selected');
            return $this->runPythonScript($uploadedImagePath, $capturedImagePath);
        } elseif ($algorithm == "fastr") {
            Log::info('Fastr algorithm selected');
            return $this->runRcnnPythonScript($uploadedImagePath, $capturedImagePath);
        } elseif ($algorithm == "yolo") {
            Log::info('Yolo algorithm selected');
            return $this->runYolocnnPythonScript($uploadedImagePath, $capturedImagePath);
        } else {
            Log::error('Invalid algorithm selected');
            return ['status' => 'error', 'message' => 'Invalid algorithm selected'];
        }
    }

    private function runPythonScript($uploadedImagePath, $capturedImagePath)
    {
        $scriptPath = realpath(base_path('scripts' . DIRECTORY_SEPARATOR . 'cnn_predict.py'));
        if (!$scriptPath) {
            Log::error('Python script not found', ['script' => $scriptPath]);
            return ['status' => 'error', 'message' => 'Python script not found'];
        }

        $command = 'python ' . escapeshellarg($scriptPath) . ' ' . escapeshellarg(public_path($uploadedImagePath)) . ' ' . escapeshellarg(public_path($capturedImagePath));
        $output = [];
        $returnVar = 0;
        exec($command . ' 2>&1', $output, $returnVar);
        $outputString = implode("\n", $output);

        $jsonString = $this->extractJsonFromOutput($outputString);
        if ($jsonString === false) {
            Log::error('Unexpected response from Python script', ['output' => $outputString]);
            return ['status' => 'error', 'message' => 'Unexpected response from the server. Please check the logs.'];
        }

        $decodedOutput = json_decode($jsonString, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            Log::error('JSON decoding error', ['output' => $jsonString, 'error' => json_last_error_msg()]);
            return ['status' => 'error', 'message' => 'Error decoding JSON response from the server.'];
        }

        Log::info('Face scan completed', ['algorithm' => 'cnn', 'output' => $decodedOutput]);

        return $decodedOutput;
    }
    private function runRcnnPythonScript($uploadedImagePath, $capturedImagePath)
    {
        $scriptPath = realpath(base_path('scripts' . DIRECTORY_SEPARATOR . 'fastr_predict.py'));
        if (!$scriptPath) {
            Log::error('Python script not found', ['script' => $scriptPath]);
            return ['status' => 'error', 'message' => 'Python script not found'];
        }

        $command = 'python ' . escapeshellarg($scriptPath) . ' ' . escapeshellarg(public_path($uploadedImagePath)) . ' ' . escapeshellarg(public_path($capturedImagePath));
        $output = [];
        $returnVar = 0;
        exec($command . ' 2>&1', $output, $returnVar);
        $outputString = implode("\n", $output);

        $jsonString = $this->extractJsonFromOutput($outputString);
        if ($jsonString === false) {
            Log::error('Unexpected response from Python script', ['output' => $outputString]);
            return ['status' => 'error', 'message' => 'Unexpected response from the server. Please check the logs.'];
        }

        $decodedOutput = json_decode($jsonString, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            Log::error('JSON decoding error', ['output' => $jsonString, 'error' => json_last_error_msg()]);
            return ['status' => 'error', 'message' => 'Error decoding JSON response from the server.'];
        }

        Log::info('Face scan completed', ['algorithm' => 'fastr', 'output' => $decodedOutput]);

        return $decodedOutput;
    }
    private function runYolocnnPythonScript($uploadedImagePath, $capturedImagePath)
    {
        $scriptPath = realpath(base_path('scripts' . DIRECTORY_SEPARATOR . 'yolo_predict.py'));
        if (!$scriptPath) {
            Log::error('Python script not found', ['script' => $scriptPath]);
            return ['status' => 'error', 'message' => 'Python script not found'];
        }

        $command = 'python ' . escapeshellarg($scriptPath) . ' ' . escapeshellarg(public_path($uploadedImagePath)) . ' ' . escapeshellarg(public_path($capturedImagePath));
        $output = [];
        $returnVar = 0;
        exec($command . ' 2>&1', $output, $returnVar);
        $outputString = implode("\n", $output);

        $jsonString = $this->extractJsonFromOutput($outputString);
        if ($jsonString === false) {
            Log::error('Unexpected response from Python script', ['output' => $outputString]);
            return ['status' => 'error', 'message' => 'Unexpected response from the server. Please check the logs.'];
        }

        $decodedOutput = json_decode($jsonString, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            Log::error('JSON decoding error', ['output' => $jsonString, 'error' => json_last_error_msg()]);
            return ['status' => 'error', 'message' => 'Error decoding JSON response from the server.'];
        }

        Log::info('Face scan completed', ['algorithm' => 'fastr', 'output' => $decodedOutput]);

        return $decodedOutput;
    }

    private function extractJsonFromOutput($outputString)
    {
        $jsonStart = strpos($outputString, '{');
        $jsonEnd = strrpos($outputString, '}');
        if ($jsonStart !== false && $jsonEnd !== false) {
            return substr($outputString, $jsonStart, $jsonEnd - $jsonStart + 1);
        }
        return false;
    }

    public function index()
    {
        return view('welcome');
    }
}

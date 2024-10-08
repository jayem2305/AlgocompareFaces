<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/webcamjs/1.0.26/webcam.min.js"></script>
    <meta name="csrf-token" content="{{ csrf_token() }}">
<style>
  
  .camera-container {
    width: 120px;  /* Adjust as needed */
    height: 200px; /* Adjust as needed */
    border-radius: 10%; /* Circular shape */
    border: 4px solid #007bff; /* Border styling */
    overflow: hidden; /* Crop overflow */
    display: flex; /* Center the webcam */
    align-items: center; /* Center vertically */
    justify-content: center; /* Center horizontally */
}
    #results-card {
        display: block;
    }
</style>

</head>
<body class="container mt-5">
<div class="row">
    <div class="col-lg-4">
        <!-- Algorithm Selection and Live Face Scanning -->
        <div class="card ">
            <div class="card-header">
                <h4>Compare Faces</h4>
            </div>
            <div class="card-body">
                <form id="scan-form" action="{{ route('face.scan') }}" method="POST">
                    @csrf
                    <div class="form-group mb-3">
                        <label for="algorithm">Choose Algorithm</label>
                        <select name="algorithm" id="algorithm" class="form-control">
                            <option value="cnn">CNN</option>
                            <option value="yolo">YOLO</option>
                            <option value="fastr">Fast-R</option>
                        </select>
                    </div>
                    <div id="dsplay_image"></div>
                    <div class="form-group mb-3">
                        <label for="image">Choose Image</label>
                        <input type="file" class="form-control" id="image" name="image" required>
                    </div>
                    <div class="form-group mb-3">
                        <label for="scan_direction">Choose Scan Direction</label><br>
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="radio" name="scan_direction" id="front_face" value="front" checked>
                            <label class="form-check-label" for="front_face">Front Face</label>
                        </div>
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="radio" name="scan_direction" id="left_side" value="left">
                            <label class="form-check-label" for="left_side">Left Side</label>
                        </div>
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="radio" name="scan_direction" id="right_side" value="right">
                            <label class="form-check-label" for="right_side">Right Side</label>
                        </div>
                    </div>
 
                    <div class="row justify-content-center">
                        <div class="col-lg-4">
                            <div class="camera-container">
                                <div id="my_camera"></div>
                            </div>
                        </div>
                    </div>
                    <br>
                    <input type="hidden" name="captured_image" id="captured_image">

                    <button type="button" id="scan-button" class="btn btn-success">Start Face Scan</button>
                    <input type="hidden" name="captured_image" id="captured_image">
                </form>
            </div>
        </div>
    </div>
    <div class="col-lg-8">
    <div class="card " id="results-card">
        <div class="card-header">
            <h4>Raw Results & Average Metrics</h4>
        </div>
        <div class="card-body">
            <h5>Raw Results:</h5>
            <table class="table table-bordered">
                <tbody>
                    <tr>
                        <th>Algorithm</th>
                        <td id="algo"></td>
                    </tr>
                    <tr>
                        <th>Uploaded Image</th>
                        <th>Image Name</th>
                        <th>Avg. Pixel Image Red Chanel</th>
                        <th>Avg. Pixel Image Green Chanel</th>
                        <th>Avg. Pixel Image Blue Chanel</th>
                        <th>Avg. Eye Distance</th>

                        
                    </tr>
                    <tr>
                        <td>
                            <div class="mt-4">
                                <img id="imageid-result" src="" alt="Uploaded Image" width="200" height="200" style="display:none;">
                            </div>
                        </td>
                        <td id="image-name">No image name available</td>
                        <td id="image-red">0</td>
                        <td id="image-green">0</td>
                        <td id="image-blue">0</td>
                        <td id="image-eye">0</td>

                    </tr>
                    <tr>
                        <th>Scanned Image</th>
                        <th>Image Name</th>
                        <th>Avg. Pixel Image Red Chanel</th>
                        <th>Avg. Pixel Image Green Chanel</th>
                        <th>Avg. Pixel Image Blue Chanel</th>
                        <th>Avg. Eye Distance</th>
                    </tr>
                    <tr>
                        <td>
                            <div class="mt-4">
                                <img id="image-result" src="../images\captured_image.jpg" alt="Scanned Image" width="200" height="200" style="display:none;">
                            </div>
                        </td>
                        <td id="image-name2">No image name available</td>
                        <td id="image-red2">0</td>
                        <td id="image-green2">0</td>
                        <td id="image-blue2">0</td>
                        <td id="image-eye2">0</td>

                    </tr>
                    <tr>
                        <th>Face Recognition</th>
                        <th>Face Matches</th>
                        <th>Similarity Score</th>
                        <th>Avg. Faces Distance</th>
                        <th>Avg. Eyes Distance</th>
                        <th>Time Interval</th>
                    </tr>
                    <tr>
                        <td id="detectface"></td>
                        <td id="face_match"></td>
                        <td id="face_similarity">0</td>
                        <td id="face_distance">0</td>
                        <td id="eye_distance">0</td>
                        <td id="timeinterval">0</td>
                        <td></td>
                    </tr>
                </tbody>
            </table>

            <h5>Average Scores:</h5>
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Precision</td>
                        <td id="avg-precision">0</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td id="avg-recall">0</td>
                    </tr>
                    <tr>
                        <td>F1 Score</td>
                        <td id="avg-f1">0</td>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td id="avg-accuracy">0</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

</div>


    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>

   <script>
        // Configure Webcam
        Webcam.set({
    width: 320,  // Keep this the same as the width and height of the CSS
    height: 320, // Set height equal to width for consistency with CSS
    image_format: 'jpeg',
    jpeg_quality: 90,
    constraints: {
        video: {
            facingMode: "user",
            width: { ideal: 320 },
            height: { ideal: 320 }  // Match the height to make the feed square
        }
    }
});

        Webcam.attach('#my_camera');
        document.getElementById('scan-button').addEventListener('click', function() {
    // Capture image from webcam
    Webcam.snap(function(data_uri) {
        // Convert the image data URI to a Blob
        fetch(data_uri)
            .then(res => res.blob())
            .then(blob => {
                // Create a new FormData object and append the captured image
                var fileInput = document.getElementById('image');
                var file = fileInput.files[0];

                if (!file) {
                    alert('Please select an image to upload.');
                    return;
                }

                // Create FormData object and append file and CSRF token
                const formData = new FormData();
                formData.append('image', file);
                formData.append('algorithm', document.getElementById('algorithm').value);
                formData.append('scan_direction', document.querySelector('input[name="scan_direction"]:checked').value);
                formData.append('captured_image', blob, 'captured_image.jpg');
                formData.append('_token', document.querySelector('meta[name="csrf-token"]').getAttribute('content'));

                // Create a new XMLHttpRequest
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '{{ route("face.scan") }}', true);

                // Set up the callback function
                xhr.onload = function () {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const data = JSON.parse(xhr.responseText);

                            const uploadedImageElement = document.getElementById('imageid-result');
                            const scannedImageElement = document.getElementById('image-result');
                            const nameElement = document.getElementById('image-name');
                            const nameElement2 = document.getElementById('image-name2');
                            const detectFaceElement = document.getElementById('detectface');
                            const algo = document.getElementById('algo');
                            const face_similarity = document.getElementById('face_similarity');
                            const face_distance = document.getElementById('face_distance');
                            const eye_distance = document.getElementById('eye_distance');
                            const timeinterval = document.getElementById('timeinterval');
                            
                            if (data.result) {
                            if (data.result.avg_pixel_values_image1) {
                                document.getElementById('image-red').textContent = data.result.avg_pixel_values_image1.Red || 'N/A';
                                document.getElementById('image-green').textContent = data.result.avg_pixel_values_image1.Green || 'N/A';
                                document.getElementById('image-blue').textContent = data.result.avg_pixel_values_image1.Blue || 'N/A';
                            }
                            if (data.result.avg_pixel_values_image2) {
                                document.getElementById('image-red2').textContent = data.result.avg_pixel_values_image2.Red || 'N/A';
                                document.getElementById('image-green2').textContent = data.result.avg_pixel_values_image2.Green || 'N/A';
                                document.getElementById('image-blue2').textContent = data.result.avg_pixel_values_image2.Blue || 'N/A';
                            }

                                document.getElementById('image-eye').textContent = data.result.average_eye_distance_image1 || 'N/A';
                                document.getElementById('image-eye2').textContent = data.result.average_eye_distance_image2 || 'N/A';

                                document.getElementById('face_match').textContent = data.result.message;
                                const algorithm = document.getElementById('algorithm').value;
                                algo.textContent = algorithm;
                                face_similarity.textContent = data.result.similarity_score;
                                face_distance.textContent = data.result.distance;
                                eye_distance.textContent = data.result.average_eye_distances;
                                timeinterval.textContent = data.result.execution_time;
                                
                                // Handle uploaded image path
                                const uploadedImagePath = data.result.image1;
                                if (uploadedImagePath) {
                                    const relativeUploadedPath = uploadedImagePath.replace(/^.*[\\\/]public[\\\/]/, '../');
                                    uploadedImageElement.src = relativeUploadedPath;
                                    uploadedImageElement.style.display = 'block';
                                    const filename1 = uploadedImagePath.split('\\').pop().split('/').pop();
                                    nameElement.textContent = `Image Name: ${filename1}`;
                                } else {
                                    uploadedImageElement.style.display = 'none';
                                    nameElement.textContent = 'No image name available';
                                }

                                // Handle scanned image path
                                const scannedImagePath = data.result.image2;
                                if (scannedImagePath) {
                                    const relativeScannedPath = scannedImagePath.replace(/^.*[\\\/]public[\\\/]/, '../');
                                    scannedImageElement.src = relativeScannedPath;
                                    scannedImageElement.style.display = 'block';
                                    const filename2 = scannedImagePath.split('\\').pop().split('/').pop();
                                    nameElement2.textContent = `Image Name: ${filename2}`;
                                } else {
                                    scannedImageElement.style.display = 'none';
                                    nameElement2.textContent = 'No image name available';
                                }

                                detectFaceElement.textContent = 'Face Detected'; // Update detection status
                            } else {
                                uploadedImageElement.style.display = 'none';
                                scannedImageElement.style.display = 'none';
                                nameElement.textContent = 'No image name available';
                                nameElement2.textContent = 'No image name available';
                                detectFaceElement.textContent = 'Face is not detected';
                            }

                            // Display average metrics
                            if (data.result.metrics) {
                                console.log('Metrics:', data.result.metrics);

                                document.getElementById('avg-precision').textContent = data.result.metrics.Precision || 'N/A';
                                document.getElementById('avg-recall').textContent = data.result.metrics.Recall || 'N/A';
                                document.getElementById('avg-f1').textContent = data.result.metrics['F1 Score'] || 'N/A';
                                document.getElementById('avg-accuracy').textContent = data.result.metrics.Accuracy || 'N/A';
                            }

                        } catch (e) {
                            console.error('Error parsing response:', e);
                        }
                    } else {
                        console.error('Request failed with status:', xhr.status);
                    }
                };

                xhr.send(formData);
            });
    });
});




    </script>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const imageInput = document.getElementById('image');
        const displayDiv = document.getElementById('dsplay_image');
        
        imageInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const imgElement = document.createElement('img');
                    imgElement.src = e.target.result;
                    imgElement.style.maxWidth = '100%'; // Adjust as needed
                    imgElement.style.maxHeight = '400px'; // Adjust as needed

                    // Clear previous content and display new image
                    displayDiv.innerHTML = '';
                    displayDiv.appendChild(imgElement);
                };
                
                reader.readAsDataURL(file);
            }
        });
    });
</script>

</body>
</html>

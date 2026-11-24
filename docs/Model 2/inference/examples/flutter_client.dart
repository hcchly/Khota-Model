// Minimal Flutter client for the Khota API (package:http).
// pubspec.yaml:  dependencies: http: ^1.2.0
import 'dart:convert';
import 'package:http/http.dart' as http;

const khotaApi = 'http://YOUR_SERVER:8000';

/// Sends one photo to the model and returns the decoded JSON result.
Future<Map<String, dynamic>> analysePhoto(String filePath) async {
  final request = http.MultipartRequest('POST', Uri.parse('$khotaApi/predict'))
    ..files.add(await http.MultipartFile.fromPath('image', filePath));
  final streamed = await request.send().timeout(const Duration(seconds: 120));
  final body = await streamed.stream.bytesToString();
  if (streamed.statusCode != 200) {
    throw Exception('Khota API error ${streamed.statusCode}: $body');
  }
  return jsonDecode(body) as Map<String, dynamic>;
}

/// Detection model only: boxes and defect type, no severity.
Future<Map<String, dynamic>> detectDefects(String filePath) async {
  final request = http.MultipartRequest('POST', Uri.parse('$khotaApi/detect'))
    ..files.add(await http.MultipartFile.fromPath('image', filePath));
  final streamed = await request.send().timeout(const Duration(seconds: 120));
  final body = await streamed.stream.bytesToString();
  if (streamed.statusCode != 200) throw Exception('Khota API error ${streamed.statusCode}: $body');
  return jsonDecode(body) as Map<String, dynamic>;
}

/// Severity model only, for boxes [[x0, y0, x1, y1], ...] in pixels of the photo.
Future<Map<String, dynamic>> rateSeverity(String filePath, List<List<num>> boxes) async {
  final request = http.MultipartRequest('POST', Uri.parse('$khotaApi/severity'))
    ..files.add(await http.MultipartFile.fromPath('image', filePath))
    ..fields['boxes'] = jsonEncode(boxes);
  final streamed = await request.send().timeout(const Duration(seconds: 120));
  final body = await streamed.stream.bytesToString();
  if (streamed.statusCode != 200) throw Exception('Khota API error ${streamed.statusCode}: $body');
  return jsonDecode(body) as Map<String, dynamic>;
}

/// Example use: list the defects and scale the boxes to the size the photo is shown at.
Future<void> example(String filePath, double shownWidth) async {
  final result = await analysePhoto(filePath);
  final scale = shownWidth / (result['width'] as num); // boxes are in pixels of the (EXIF-rotated) photo
  for (final d in (result['defects'] as List)) {
    final box = (d['box_xyxy'] as List).map((v) => (v as num) * scale).toList();
    print('${d['class']} | ${d['severity']} | conf ${d['confidence']} | box $box');
  }
  if ((result['defects'] as List).isEmpty) print(result['result']); // "No Target Defect"
}

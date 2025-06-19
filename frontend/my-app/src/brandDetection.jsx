import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Grid,
  Paper,
  TextField,
  Typography,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  CloudUpload,
  PhotoCamera,
  Send,
  CheckCircle,
  Error,
  Delete as DeleteIcon,
  Add,
  Download as DownloadIcon
} from '@mui/icons-material';
import JSZip from 'jszip';

const BrandDetection = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [brands, setBrands] = useState([
    { name: '', variations: '' }  // Initial empty brand
  ]);
  const [productDescription, setProductDescription] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [processedFiles, setProcessedFiles] = useState({});

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      setSelectedFiles(files);
      const urls = files.map(file => URL.createObjectURL(file));
      setPreviewUrls(urls);
      setError('');
      setResults([]);
    }
  };

  const handleRemoveImage = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    setPreviewUrls(prev => {
      const newUrls = [...prev];
      URL.revokeObjectURL(newUrls[index]); // Clean up the URL
      return newUrls.filter((_, i) => i !== index);
    });
  };

  const handleAddBrand = () => {
    setBrands([...brands, { name: '', variations: '' }]);
  };

  const handleRemoveBrand = (index) => {
    setBrands(brands.filter((_, i) => i !== index));
  };

  const handleBrandChange = (index, field, value) => {
    const newBrands = [...brands];
    newBrands[index][field] = value;
    setBrands(newBrands);
  };

  const handleSubmit = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image file');
      return;
    }

    // Validate brands
    const validBrands = brands.filter(brand => brand.name.trim());
    if (validBrands.length === 0) {
      setError('Please enter at least one brand name');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const allResults = [];
      const newProcessedFiles = {};
      
      for (const file of selectedFiles) {
        const formData = new FormData();
        formData.append('image', file);
        
        // Prepare brand data
        const brandData = validBrands.map(brand => ({
          name: brand.name.trim(),
          aliases: brand.variations.split(',').map(v => v.trim()).filter(v => v),
          description: productDescription,
          product_types: productDescription.split(',').map(v => v.trim()).filter(v => v)
        }));
        
        formData.append('brands', JSON.stringify(brandData));

        const response = await fetch('http://localhost:8000/api/detect', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        allResults.push({
          fileName: file.name,
          result: data
        });

        // Store the processed file information
        if (data.renamed_file) {
          newProcessedFiles[file.name] = {
            originalFile: file,
            newName: data.renamed_file.new_name
          };
        }
      }
      
      setResults(allResults);
      setProcessedFiles(newProcessedFiles);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const zip = new JSZip();
      
      // Add each processed file to the zip
      for (const [originalName, fileInfo] of Object.entries(processedFiles)) {
        zip.file(fileInfo.newName, fileInfo.originalFile);
      }
      
      // Generate and download the zip file
      const content = await zip.generateAsync({ type: "blob" });
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.download = "processed_images.zip";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Error creating zip file: ${err.message}`);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceText = (confidence) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
        Brand & Product Detection
      </Typography>
      
      <Grid container spacing={4}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Upload & Configure
              </Typography>
              
              {/* File Upload */}
              <Box sx={{ mb: 3 }}>
                <input
                  accept="image/*"
                  style={{ display: 'none' }}
                  id="image-upload"
                  type="file"
                  multiple
                  onChange={handleFileSelect}
                />
                <label htmlFor="image-upload">
                  <Button
                    variant="outlined"
                    component="span"
                    startIcon={<CloudUpload />}
                    fullWidth
                    sx={{ mb: 2, py: 2 }}
                  >
                    Select Images
                  </Button>
                </label>
                
                {previewUrls.length > 0 && (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
                    {previewUrls.map((url, index) => (
                      <Box key={index} sx={{ position: 'relative' }}>
                        <img
                          src={url}
                          alt={`Preview ${index + 1}`}
                          style={{ 
                            width: '100px',
                            height: '100px',
                            objectFit: 'cover',
                            borderRadius: '8px',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                          }}
                        />
                        <IconButton
                          size="small"
                          onClick={() => handleRemoveImage(index)}
                          sx={{
                            position: 'absolute',
                            top: -8,
                            right: -8,
                            backgroundColor: 'white',
                            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                            '&:hover': {
                              backgroundColor: 'grey.100'
                            }
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                        <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                          {selectedFiles[index]?.name}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </Box>

              {/* Brand Configuration */}
              <Paper sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Brand Configuration
                </Typography>
                
                {brands.map((brand, index) => (
                  <Box key={index} sx={{ mb: 2, position: 'relative' }}>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={11}>
                        <TextField
                          label="Brand Name"
                          value={brand.name}
                          onChange={(e) => handleBrandChange(index, 'name', e.target.value)}
                          fullWidth
                          size="small"
                          sx={{ mb: 1 }}
                        />
                        <TextField
                          label="Brand Variations (comma-separated)"
                          value={brand.variations}
                          onChange={(e) => handleBrandChange(index, 'variations', e.target.value)}
                          fullWidth
                          size="small"
                          helperText="Enter different variations of this brand name, separated by commas"
                        />
                      </Grid>
                      <Grid item xs={1}>
                        <IconButton 
                          onClick={() => handleRemoveBrand(index)}
                          disabled={brands.length === 1}
                          sx={{ mt: 1 }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Grid>
                    </Grid>
                  </Box>
                ))}
                
                <Button
                  variant="outlined"
                  onClick={handleAddBrand}
                  startIcon={<Add />}
                  sx={{ mt: 1 }}
                >
                  Add Another Brand
                </Button>

                <TextField
                  label="Product Description"
                  value={productDescription}
                  onChange={(e) => setProductDescription(e.target.value)}
                  fullWidth
                  multiline
                  rows={3}
                  size="small"
                  sx={{ mt: 2 }}
                  helperText="Describe the types of products these brands sell"
                />
              </Paper>

              {/* Submit Button */}
              <Button
                variant="contained"
                onClick={handleSubmit}
                disabled={loading || selectedFiles.length === 0}
                startIcon={loading ? <CircularProgress size={20} /> : <Send />}
                fullWidth
                size="large"
                sx={{ mt: 2 }}
              >
                {loading ? 'Processing...' : 'Detect Brand & Product'}
              </Button>

              {/* Error Display */}
              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h5">
                  Detection Results
                </Typography>
                {Object.keys(processedFiles).length > 0 && (
                  <Tooltip title="Download processed images">
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<DownloadIcon />}
                      onClick={handleDownload}
                    >
                      Download
                    </Button>
                  </Tooltip>
                )}
              </Box>
              
              {loading && (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <CircularProgress size={60} />
                  <Typography variant="body1" sx={{ mt: 2 }}>
                    Analyzing images...
                  </Typography>
                </Box>
              )}

              {results.map((result, index) => (
                <Box key={index} sx={{ mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    {result.fileName}
                    {result.result.renamed_file && (
                      <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                        → {result.result.renamed_file.new_name}
                      </Typography>
                    )}
                  </Typography>
                  
                  <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={8}>
                        <Typography variant="h6" color="primary">
                          {result.result.brand}
                        </Typography>
                        <Typography variant="body1" color="text.secondary">
                          {result.result.best_product || "Unknown Product"}
                        </Typography>
                        {result.result.processing_metadata.explanation && (
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                            {result.result.processing_metadata.explanation}
                          </Typography>
                        )}
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'right' }}>
                          <Chip
                            label={`${Math.round(result.result.confidence * 100)}%`}
                            color={getConfidenceColor(result.result.confidence)}
                            icon={result.result.confidence >= 0.6 ? <CheckCircle /> : <Error />}
                          />
                          <Typography variant="caption" display="block">
                            {getConfidenceText(result.result.confidence)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                            Using {result.result.processing_metadata.product_detection_method}
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                    
                    <LinearProgress 
                      variant="determinate" 
                      value={result.result.confidence * 100}
                      sx={{ mt: 1 }}
                      color={getConfidenceColor(result.result.confidence)}
                    />
                  </Paper>

                  {/* Extracted Text */}
                  {result.result.extracted_text && (
                    <>
                      <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
                        Extracted Text
                      </Typography>
                      <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
                        <Typography variant="body2" style={{ whiteSpace: 'pre-wrap' }}>
                          {result.result.extracted_text}
                        </Typography>
                      </Paper>
                    </>
                  )}
                  
                  {index < results.length - 1 && <Divider sx={{ my: 3 }} />}
                </Box>
              ))}

              {!loading && results.length === 0 && !error && (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <PhotoCamera sx={{ fontSize: 60, color: 'grey.400', mb: 2 }} />
                  <Typography variant="body1" color="text.secondary">
                    Upload images to get started
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default BrandDetection;
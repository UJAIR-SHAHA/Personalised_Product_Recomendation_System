function viewSimilar(productId) {
    if (!productId) {
        console.error("Product ID is undefined!");
        return;
    }
    // Correct the concatenation syntax
    var url = '/View_Similar_Product?product_id=' + productId;
    console.log('Navigating to:', url); // Debugging log
    window.location.href = url; // Navigate to the Flask route
}
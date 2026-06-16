const path = require("path");

// Load the compiled native addon
const addonPath = path.join(__dirname, "build", "Release", "octrajs.node");
const octrajs = require(addonPath);

// Export the addon or wrap it as needed
module.exports = octrajs;

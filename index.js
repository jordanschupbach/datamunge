const path = require("path");

// Load the compiled native addon
const addonPath = path.join(__dirname, "build", "Release", "datamungejs.node");
const datamungejs = require(addonPath);

// Export the addon or wrap it as needed
module.exports = datamungejs;

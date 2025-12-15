const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const { ModuleFederationPlugin } = require("webpack").container;
const deps = require("./package.json").dependencies;

// BrainDrive RAG plugin federation settings
const PLUGIN_SCOPE = "BrainDriveRAGCommunity";
const MODULE_NAME = "BrainDriveRAGSettings";
const DEV_PORT = 3015;

module.exports = {
  mode: "development",
  entry: "./src/index",
  output: {
    //path: path.resolve(__dirname, "dist"),
    path: path.resolve(__dirname, "../../backend/plugins/shared/BrainDriveRAGCommunity/v1.0.0/dist"),
    publicPath: "auto",
    clean: true,
    library: {
      type: "var",
      name: PLUGIN_SCOPE
    }
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"]
  },
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/,
        use: "ts-loader",
        exclude: /node_modules/
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"]
      }
    ]
  },
  plugins: [
    new ModuleFederationPlugin({
      name: PLUGIN_SCOPE,
      library: { type: "var", name: PLUGIN_SCOPE },
      filename: "remoteEntry.js",
      exposes: {
        [`./${MODULE_NAME}`]: "./src/RAGSettingsPanel"
      },
      shared: {
        react: { singleton: true, requiredVersion: deps.react, eager: true },
        "react-dom": { singleton: true, requiredVersion: deps["react-dom"], eager: true }
      }
    }),
    new HtmlWebpackPlugin({
      template: "./public/index.html"
    })
  ],
  devServer: {
    port: DEV_PORT,
    static: {
      directory: path.join(__dirname, "public")
    },
    hot: true
  }
};

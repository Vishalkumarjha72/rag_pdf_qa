# ---- Stage 1: build ----
# Use a Node image only to install deps and produce the static build output.
# This entire stage (Node, node_modules, source files) is discarded after
# the build — it never ships in the final image.
FROM node:20-alpine AS build

WORKDIR /app

# Copy only the manifest files first so Docker can cache the npm install
# layer — it only re-runs if package.json/package-lock.json actually change,
# same caching trick we used for pip install in the backend Dockerfile.
COPY package.json package-lock.json ./
RUN npm install

# Now copy the rest of the source and build the static bundle.
# Vite outputs plain HTML/CSS/JS into /app/dist.
COPY . .
RUN npm run build

# ---- Stage 2: serve ----
# Start fresh from a tiny nginx image — no Node, no node_modules, no source.
# Only the built static files get copied in from the previous stage.
FROM nginx:alpine

# Replace the default nginx welcome page with our built frontend.
COPY --from=build /app/dist /usr/share/nginx/html

# Nginx's default config already serves on port 80 and works for a
# single-page app's static assets, so no custom nginx.conf needed yet.
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

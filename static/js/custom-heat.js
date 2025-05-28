/* Custom Heatmap Layer with optimized canvas handling */
L.HeatLayer = L.Layer.extend({
    initialize: function (latlngs, options) {
        this._latlngs = latlngs;
        L.setOptions(this, options);
    },

    setLatLngs: function (latlngs) {
        this._latlngs = latlngs;
        return this.redraw();
    },

    addLatLng: function (latlng) {
        this._latlngs.push(latlng);
        return this.redraw();
    },

    setOptions: function (options) {
        L.setOptions(this, options);
        if (this._canvas) {
            this._updateOptions();
        }
        return this.redraw();
    },

    _updateOptions: function() {
        if (this.options.opacity !== undefined) {
            this._canvas.style.opacity = this.options.opacity;
        }
    },

    redraw: function () {
        if (!this._frame && this._map && !this._map._animating) {
            this._frame = L.Util.requestAnimFrame(this._redraw, this);
        }
        return this;
    },

    onAdd: function (map) {
        this._map = map;

        if (!this._canvas) {
            this._initCanvas();
        }

        if (this.options.pane) {
            this.getPane().appendChild(this._canvas);
        } else {
            map._panes.overlayPane.appendChild(this._canvas);
        }

        map.on('moveend', this._reset, this);

        if (map.options.zoomAnimation && L.Browser.any3d) {
            map.on('zoomanim', this._animateZoom, this);
        }

        this._reset();
    },

    onRemove: function (map) {
        if (this.options.pane) {
            this.getPane().removeChild(this._canvas);
        } else {
            map.getPanes().overlayPane.removeChild(this._canvas);
        }

        map.off('moveend', this._reset, this);

        if (map.options.zoomAnimation) {
            map.off('zoomanim', this._animateZoom, this);
        }
    },

    _initCanvas: function () {
        let canvas = L.DomUtil.create('canvas', 'leaflet-heatmap-layer leaflet-layer');

        this._canvas = canvas;
        this._ctx = canvas.getContext('2d', {
            willReadFrequently: true
        });

        const size = this._map.getSize();
        canvas.width = size.x;
        canvas.height = size.y;

        const animated = this._map.options.zoomAnimation && L.Browser.any3d;
        L.DomUtil.addClass(canvas, 'leaflet-zoom-' + (animated ? 'animated' : 'hide'));

        if (this.options.opacity !== undefined) {
            canvas.style.opacity = this.options.opacity;
        }
    },

    _reset: function () {
        const topLeft = this._map.containerPointToLayerPoint([0, 0]);
        L.DomUtil.setPosition(this._canvas, topLeft);

        const size = this._map.getSize();

        if (this._canvas.width !== size.x) {
            this._canvas.width = size.x;
        }
        if (this._canvas.height !== size.y) {
            this._canvas.height = size.y;
        }

        this._redraw();
    },

    _redraw: function () {
        if (!this._map) {
            return;
        }

        const data = [];
        const r = this._canvas.height / 2;
        const size = this._map.getSize();
        const bounds = new L.Bounds(
            L.point([-r, -r]),
            size.add([r, r]));

        // Clear canvas
        this._ctx.clearRect(0, 0, this._canvas.width, this._canvas.height);

        // Convert lat/lng points to pixels and add to data array
        for (let i = 0, len = this._latlngs.length; i < len; i++) {
            const p = this._map.latLngToContainerPoint(this._latlngs[i]);
            if (bounds.contains(p)) {
                data.push([
                    Math.round(p.x),
                    Math.round(p.y),
                    this._latlngs[i][2]
                ]);
            }
        }

        // Draw gradient
        const gradient = this._ctx.createLinearGradient(0, 0, 0, 256);
        for (let key in this.options.gradient) {
            gradient.addColorStop(key, this.options.gradient[key]);
        }

        // Draw heatmap points
        this._ctx.globalAlpha = this.options.minOpacity || 0.05;

        for (let i = 0, len = data.length; i < len; i++) {
            const p = data[i];
            const intensity = Math.min(1, Math.max(p[2], 0));
            
            this._ctx.beginPath();
            this._ctx.fillStyle = gradient;
            this._ctx.arc(p[0], p[1], this.options.radius || 25, 0, Math.PI * 2, false);
            this._ctx.fill();
        }

        this._frame = null;
    },

    _animateZoom: function (e) {
        const scale = this._map.getZoomScale(e.zoom);
        const offset = this._map._getCenterOffset(e.center)._multiplyBy(-scale).subtract(this._map._getMapPanePos());

        if (L.DomUtil.setTransform) {
            L.DomUtil.setTransform(this._canvas, offset, scale);
        } else {
            this._canvas.style[L.DomUtil.TRANSFORM] = L.DomUtil.getTranslateString(offset) + ' scale(' + scale + ')';
        }
    }
}); 
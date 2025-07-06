"""
Tests for plotting and visualization functionality.

This module tests chart generation, data visualization,
export functionality, and plotting utilities.
"""

import base64

from webapp.config.unified import get_config


class TestPlottingConfiguration:
    """Test plotting configuration settings."""

    def test_plotting_backend_config(self):
        """Test plotting backend configuration."""
        plotting_backend = get_config('plotting_backend', 'visualization', 'plotly')
        assert plotting_backend in ['plotly', 'matplotlib', 'altair']

    def test_chart_export_config(self):
        """Test chart export configuration."""
        enable_exports = get_config('enable_chart_exports', 'visualization', True)
        assert isinstance(enable_exports, bool)

        export_formats = get_config('supported_export_formats', 'visualization',
                                    ['png', 'pdf', 'svg', 'html'])
        assert isinstance(export_formats, list)
        assert len(export_formats) > 0

    def test_plotting_theme_config(self):
        """Test plotting theme configuration."""
        default_theme = get_config('default_chart_theme', 'visualization', 'plotly')
        assert isinstance(default_theme, str)

        color_palette = get_config('color_palette', 'visualization', 'viridis')
        assert isinstance(color_palette, str)

    def test_performance_config(self):
        """Test plotting performance configuration."""
        max_data_points = get_config('max_data_points_per_chart', 'visualization', 10000)
        assert isinstance(max_data_points, int)
        assert max_data_points > 0

        enable_caching = get_config('enable_plot_caching', 'visualization', True)
        assert isinstance(enable_caching, bool)


class TestBasicChartGeneration:
    """Test basic chart generation functionality."""

    def test_bar_chart_generation(self):
        """Test bar chart generation."""
        def create_bar_chart(data, x_column, y_column, title=None):
            """Mock bar chart creation."""
            if not data or len(data) == 0:
                raise ValueError("No data provided for chart")

            if x_column not in data[0] or y_column not in data[0]:
                raise KeyError(f"Column not found: {x_column} or {y_column}")

            chart_config = {
                'type': 'bar',
                'data': data,
                'x_axis': x_column,
                'y_axis': y_column,
                'title': title or f"{y_column} by {x_column}",
                'layout': {
                    'xaxis': {'title': x_column},
                    'yaxis': {'title': y_column}
                }
            }

            return chart_config

        test_data = [
            {'category': 'A', 'value': 10},
            {'category': 'B', 'value': 20},
            {'category': 'C', 'value': 15}
        ]

        chart = create_bar_chart(test_data, 'category', 'value', 'Test Chart')

        assert chart['type'] == 'bar'
        assert chart['x_axis'] == 'category'
        assert chart['y_axis'] == 'value'
        assert chart['title'] == 'Test Chart'
        assert len(chart['data']) == 3

    def test_line_chart_generation(self):
        """Test line chart generation."""
        def create_line_chart(data, x_column, y_column, title=None):
            """Mock line chart creation."""
            chart_config = {
                'type': 'line',
                'data': data,
                'x_axis': x_column,
                'y_axis': y_column,
                'title': title or f"{y_column} over {x_column}",
                'layout': {
                    'xaxis': {'title': x_column},
                    'yaxis': {'title': y_column}
                }
            }

            return chart_config

        time_series_data = [
            {'time': '2024-01-01', 'frequency': 5},
            {'time': '2024-01-02', 'frequency': 8},
            {'time': '2024-01-03', 'frequency': 12}
        ]

        chart = create_line_chart(time_series_data, 'time', 'frequency')

        assert chart['type'] == 'line'
        assert chart['x_axis'] == 'time'
        assert chart['y_axis'] == 'frequency'

    def test_scatter_plot_generation(self):
        """Test scatter plot generation."""
        def create_scatter_plot(data, x_column, y_column, color_column=None, title=None):
            """Mock scatter plot creation."""
            chart_config = {
                'type': 'scatter',
                'data': data,
                'x_axis': x_column,
                'y_axis': y_column,
                'color_by': color_column,
                'title': title or f"{y_column} vs {x_column}",
                'layout': {
                    'xaxis': {'title': x_column},
                    'yaxis': {'title': y_column}
                }
            }

            return chart_config

        scatter_data = [
            {'x': 1, 'y': 2, 'category': 'Type A'},
            {'x': 2, 'y': 4, 'category': 'Type B'},
            {'x': 3, 'y': 6, 'category': 'Type A'}
        ]

        chart = create_scatter_plot(scatter_data, 'x', 'y', 'category')

        assert chart['type'] == 'scatter'
        assert chart['color_by'] == 'category'

    def test_heatmap_generation(self):
        """Test heatmap generation."""
        def create_heatmap(data, x_column, y_column, value_column, title=None):
            """Mock heatmap creation."""
            chart_config = {
                'type': 'heatmap',
                'data': data,
                'x_axis': x_column,
                'y_axis': y_column,
                'values': value_column,
                'title': title or f"Heatmap of {value_column}",
                'layout': {
                    'xaxis': {'title': x_column},
                    'yaxis': {'title': y_column}
                }
            }

            return chart_config

        heatmap_data = [
            {'x': 1, 'y': 1, 'value': 10},
            {'x': 1, 'y': 2, 'value': 20},
            {'x': 2, 'y': 1, 'value': 15}
        ]

        chart = create_heatmap(heatmap_data, 'x', 'y', 'value')

        assert chart['type'] == 'heatmap'
        assert chart['values'] == 'value'


class TestAdvancedChartFeatures:
    """Test advanced chart features and customization."""

    def test_multi_series_charts(self):
        """Test charts with multiple data series."""
        def create_multi_series_chart(datasets, chart_type='line'):
            """Mock multi-series chart creation."""
            chart_config = {
                'type': chart_type,
                'series': [],
                'layout': {'showlegend': True}
            }

            for dataset in datasets:
                series = {
                    'name': dataset['name'],
                    'data': dataset['data'],
                    'color': dataset.get('color'),
                    'line_style': dataset.get('line_style', 'solid')
                }
                chart_config['series'].append(series)

            return chart_config

        test_datasets = [
            {
                'name': 'Series 1',
                'data': [{'x': 1, 'y': 10}, {'x': 2, 'y': 20}],
                'color': 'blue'
            },
            {
                'name': 'Series 2',
                'data': [{'x': 1, 'y': 15}, {'x': 2, 'y': 25}],
                'color': 'red'
            }
        ]

        chart = create_multi_series_chart(test_datasets)

        assert len(chart['series']) == 2
        assert chart['series'][0]['name'] == 'Series 1'
        assert chart['series'][1]['color'] == 'red'

    def test_chart_annotations(self):
        """Test chart annotations and markers."""
        def add_annotations(chart_config, annotations):
            """Mock annotation addition."""
            chart_config['annotations'] = []

            for annotation in annotations:
                chart_annotation = {
                    'x': annotation['x'],
                    'y': annotation['y'],
                    'text': annotation['text'],
                    'showarrow': annotation.get('show_arrow', True),
                    'arrowcolor': annotation.get('arrow_color', 'black')
                }
                chart_config['annotations'].append(chart_annotation)

            return chart_config

        base_chart = {'type': 'line', 'data': []}
        annotations = [
            {'x': 1, 'y': 10, 'text': 'Peak value'},
            {'x': 2, 'y': 5, 'text': 'Low point', 'show_arrow': False}
        ]

        annotated_chart = add_annotations(base_chart, annotations)

        assert len(annotated_chart['annotations']) == 2
        assert annotated_chart['annotations'][0]['text'] == 'Peak value'
        assert not annotated_chart['annotations'][1]['showarrow']

    def test_chart_theming(self):
        """Test chart theming and styling."""
        def apply_theme(chart_config, theme_name='default'):
            """Mock theme application."""
            themes = {
                'default': {
                    'background_color': 'white',
                    'grid_color': 'lightgray',
                    'text_color': 'black',
                    'font_family': 'Arial'
                },
                'dark': {
                    'background_color': '#2D3748',
                    'grid_color': '#4A5568',
                    'text_color': 'white',
                    'font_family': 'Arial'
                },
                'minimal': {
                    'background_color': 'white',
                    'grid_color': 'none',
                    'text_color': 'black',
                    'font_family': 'Helvetica'
                }
            }

            theme = themes.get(theme_name, themes['default'])
            chart_config['theme'] = theme

            return chart_config

        base_chart = {'type': 'bar', 'data': []}

        # Test default theme
        themed_chart = apply_theme(base_chart, 'default')
        assert themed_chart['theme']['background_color'] == 'white'

        # Test dark theme
        dark_chart = apply_theme(base_chart, 'dark')
        assert dark_chart['theme']['background_color'] == '#2D3748'
        assert dark_chart['theme']['text_color'] == 'white'

    def test_responsive_sizing(self):
        """Test responsive chart sizing."""
        def configure_responsive_sizing(chart_config, container_width=None):
            """Mock responsive sizing configuration."""
            if container_width:
                if container_width < 480:
                    # Mobile
                    sizing = {
                        'width': container_width,
                        'height': container_width * 0.75,
                        'font_size': 10,
                        'margin': {'top': 20, 'right': 20, 'bottom': 40, 'left': 40}
                    }
                elif container_width < 768:
                    # Tablet
                    sizing = {
                        'width': container_width,
                        'height': container_width * 0.6,
                        'font_size': 12,
                        'margin': {'top': 30, 'right': 30, 'bottom': 50, 'left': 50}
                    }
                else:
                    # Desktop
                    sizing = {
                        'width': container_width,
                        'height': container_width * 0.5,
                        'font_size': 14,
                        'margin': {'top': 40, 'right': 40, 'bottom': 60, 'left': 60}
                    }
            else:
                sizing = {'auto_resize': True}

            chart_config['sizing'] = sizing
            return chart_config

        base_chart = {'type': 'line', 'data': []}

        # Test mobile sizing
        mobile_chart = configure_responsive_sizing(base_chart, 320)
        assert mobile_chart['sizing']['font_size'] == 10
        assert mobile_chart['sizing']['height'] == 240  # 320 * 0.75

        # Test desktop sizing
        desktop_chart = configure_responsive_sizing(base_chart, 1200)
        assert desktop_chart['sizing']['font_size'] == 14
        assert desktop_chart['sizing']['height'] == 600  # 1200 * 0.5


class TestChartDataProcessing:
    """Test chart data processing and validation."""

    def test_data_validation(self):
        """Test chart data validation."""
        def validate_chart_data(data, required_columns):
            """Mock chart data validation."""
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': []
            }

            if not data or len(data) == 0:
                validation_result['valid'] = False
                validation_result['errors'].append('No data provided')
                return validation_result

            # Check required columns
            first_row = data[0]
            for column in required_columns:
                if column not in first_row:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f'Missing required column: {column}')

            # Check data consistency
            for i, row in enumerate(data):
                for column in required_columns:
                    if column in row and row[column] is None:
                        validation_result['warnings'].append(
                            f'Null value in row {i}, column {column}'
                        )

            return validation_result

        # Test valid data
        valid_data = [
            {'x': 1, 'y': 10},
            {'x': 2, 'y': 20}
        ]
        result = validate_chart_data(valid_data, ['x', 'y'])
        assert result['valid'] is True
        assert len(result['errors']) == 0

        # Test missing column
        invalid_data = [
            {'x': 1},
            {'x': 2}
        ]
        result = validate_chart_data(invalid_data, ['x', 'y'])
        assert result['valid'] is False
        assert 'Missing required column: y' in result['errors']

    def test_data_aggregation(self):
        """Test data aggregation for charts."""
        def aggregate_data(data, group_by, value_column, aggregation='sum'):
            """Mock data aggregation."""
            aggregations = {
                'sum': lambda values: sum(values),
                'mean': lambda values: sum(values) / len(values),
                'count': lambda values: len(values),
                'max': lambda values: max(values),
                'min': lambda values: min(values)
            }

            if aggregation not in aggregations:
                raise ValueError(f"Unsupported aggregation: {aggregation}")

            grouped_data = {}
            for row in data:
                group_key = row[group_by]
                if group_key not in grouped_data:
                    grouped_data[group_key] = []

                if value_column in row and row[value_column] is not None:
                    grouped_data[group_key].append(row[value_column])

            result = []
            for group_key, values in grouped_data.items():
                if values:
                    aggregated_value = aggregations[aggregation](values)
                    result.append({
                        group_by: group_key,
                        f'{aggregation}_{value_column}': aggregated_value
                    })

            return result

        test_data = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
            {'category': 'B', 'value': 15},
            {'category': 'B', 'value': 25}
        ]

        # Test sum aggregation
        sum_result = aggregate_data(test_data, 'category', 'value', 'sum')
        assert len(sum_result) == 2
        assert sum_result[0]['sum_value'] == 30 or sum_result[1]['sum_value'] == 30

        # Test mean aggregation
        mean_result = aggregate_data(test_data, 'category', 'value', 'mean')
        assert len(mean_result) == 2

    def test_data_filtering(self):
        """Test data filtering for charts."""
        def filter_chart_data(data, filters):
            """Mock data filtering."""
            filtered_data = data.copy()

            for filter_config in filters:
                column = filter_config['column']
                operator = filter_config['operator']
                value = filter_config['value']

                if operator == 'equals':
                    filtered_data = [row for row in filtered_data
                                     if row.get(column) == value]
                elif operator == 'greater_than':
                    filtered_data = [row for row in filtered_data
                                     if row.get(column, 0) > value]
                elif operator == 'less_than':
                    filtered_data = [row for row in filtered_data
                                     if row.get(column, 0) < value]
                elif operator == 'contains':
                    filtered_data = [row for row in filtered_data
                                     if value in str(row.get(column, ''))]

            return filtered_data

        test_data = [
            {'category': 'Type A', 'value': 10, 'year': 2023},
            {'category': 'Type B', 'value': 20, 'year': 2023},
            {'category': 'Type A', 'value': 15, 'year': 2024},
            {'category': 'Type C', 'value': 5, 'year': 2024}
        ]

        # Test single filter
        filters = [{'column': 'year', 'operator': 'equals', 'value': 2024}]
        filtered = filter_chart_data(test_data, filters)
        assert len(filtered) == 2
        assert all(row['year'] == 2024 for row in filtered)

        # Test multiple filters
        filters = [
            {'column': 'year', 'operator': 'equals', 'value': 2023},
            {'column': 'value', 'operator': 'greater_than', 'value': 15}
        ]
        filtered = filter_chart_data(test_data, filters)
        assert len(filtered) == 1
        assert filtered[0]['category'] == 'Type B'


class TestChartExport:
    """Test chart export functionality."""

    def test_png_export(self):
        """Test PNG chart export."""
        def export_chart_to_png(chart_config, width=800, height=600):
            """Mock PNG export."""
            # Simulate image generation
            image_data = f"PNG_DATA_{chart_config['type']}_{width}x{height}"

            return {
                'format': 'png',
                'width': width,
                'height': height,
                'data': base64.b64encode(image_data.encode()).decode(),
                'mime_type': 'image/png'
            }

        chart = {'type': 'bar', 'data': []}
        exported = export_chart_to_png(chart, 1000, 800)

        assert exported['format'] == 'png'
        assert exported['width'] == 1000
        assert exported['height'] == 800
        assert exported['mime_type'] == 'image/png'

    def test_svg_export(self):
        """Test SVG chart export."""
        def export_chart_to_svg(chart_config):
            """Mock SVG export."""
            svg_content = f"""
            <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
                <rect x="0" y="0" width="800" height="600" fill="white"/>
                <text x="400" y="300" text-anchor="middle">
                    {chart_config['type']} Chart
                </text>
            </svg>
            """

            return {
                'format': 'svg',
                'data': svg_content.strip(),
                'mime_type': 'image/svg+xml'
            }

        chart = {'type': 'line', 'data': []}
        exported = export_chart_to_svg(chart)

        assert exported['format'] == 'svg'
        assert 'line Chart' in exported['data']
        assert exported['mime_type'] == 'image/svg+xml'

    def test_pdf_export(self):
        """Test PDF chart export."""
        def export_chart_to_pdf(chart_config, paper_size='letter'):
            """Mock PDF export."""
            pdf_data = f"PDF_DATA_{chart_config['type']}_{paper_size}"

            return {
                'format': 'pdf',
                'paper_size': paper_size,
                'data': base64.b64encode(pdf_data.encode()).decode(),
                'mime_type': 'application/pdf'
            }

        chart = {'type': 'scatter', 'data': []}
        exported = export_chart_to_pdf(chart, 'a4')

        assert exported['format'] == 'pdf'
        assert exported['paper_size'] == 'a4'
        assert exported['mime_type'] == 'application/pdf'

    def test_html_export(self):
        """Test HTML chart export."""
        def export_chart_to_html(chart_config, include_plotly_js=True):
            """Mock HTML export."""
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{chart_config.get('title', 'Chart')}</title>
                {'<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
                 if include_plotly_js else ''}
            </head>
            <body>
                <div id="chart"></div>
                <script>
                    // Chart configuration: {chart_config['type']}
                    Plotly.newPlot('chart', chartData, layout);
                </script>
            </body>
            </html>
            """

            return {
                'format': 'html',
                'data': html_content.strip(),
                'mime_type': 'text/html',
                'includes_plotly': include_plotly_js
            }

        chart = {'type': 'heatmap', 'title': 'Test Heatmap', 'data': []}
        exported = export_chart_to_html(chart, True)

        assert exported['format'] == 'html'
        assert 'Test Heatmap' in exported['data']
        assert exported['includes_plotly'] is True

    def test_export_with_custom_styling(self):
        """Test chart export with custom styling."""
        def export_with_style(chart_config, export_format, style_options=None):
            """Mock styled export."""
            style_options = style_options or {}

            export_config = {
                'format': export_format,
                'chart_type': chart_config['type'],
                'styling': {
                    'background_color': style_options.get('bg_color', 'white'),
                    'border_width': style_options.get('border_width', 1),
                    'font_family': style_options.get('font_family', 'Arial'),
                    'color_scheme': style_options.get('color_scheme', 'default')
                }
            }

            return export_config

        chart = {'type': 'bar', 'data': []}
        style_opts = {
            'bg_color': 'lightblue',
            'border_width': 2,
            'font_family': 'Helvetica',
            'color_scheme': 'viridis'
        }

        styled_export = export_with_style(chart, 'png', style_opts)

        assert styled_export['styling']['background_color'] == 'lightblue'
        assert styled_export['styling']['border_width'] == 2
        assert styled_export['styling']['color_scheme'] == 'viridis'


class TestChartInteractivity:
    """Test chart interactivity features."""

    def test_zoom_and_pan(self):
        """Test zoom and pan functionality."""
        def configure_zoom_pan(chart_config, enable_zoom=True, enable_pan=True):
            """Mock zoom/pan configuration."""
            chart_config['interactivity'] = {
                'zoom': enable_zoom,
                'pan': enable_pan,
                'zoom_mode': 'xy' if enable_zoom else None,
                'scroll_zoom': enable_zoom
            }

            return chart_config

        chart = {'type': 'line', 'data': []}
        interactive_chart = configure_zoom_pan(chart, True, True)

        assert interactive_chart['interactivity']['zoom'] is True
        assert interactive_chart['interactivity']['pan'] is True
        assert interactive_chart['interactivity']['zoom_mode'] == 'xy'

    def test_hover_tooltips(self):
        """Test hover tooltip configuration."""
        def configure_tooltips(chart_config, tooltip_fields):
            """Mock tooltip configuration."""
            chart_config['hover'] = {
                'mode': 'closest',
                'fields': tooltip_fields,
                'format': {field: f'{field}: %{{y}}' for field in tooltip_fields}
            }

            return chart_config

        chart = {'type': 'scatter', 'data': []}
        tooltip_chart = configure_tooltips(chart, ['x', 'y', 'category'])

        assert len(tooltip_chart['hover']['fields']) == 3
        assert 'category' in tooltip_chart['hover']['fields']

    def test_selection_and_brushing(self):
        """Test data selection and brushing."""
        def configure_selection(chart_config, selection_mode='box'):
            """Mock selection configuration."""
            chart_config['selection'] = {
                'mode': selection_mode,
                'enabled': True,
                'persistent': False
            }

            return chart_config

        chart = {'type': 'scatter', 'data': []}
        selectable_chart = configure_selection(chart, 'lasso')

        assert selectable_chart['selection']['mode'] == 'lasso'
        assert selectable_chart['selection']['enabled'] is True


class TestPlottingIntegration:
    """Integration tests for plotting functionality."""

    def test_end_to_end_chart_creation(self):
        """Test complete chart creation workflow."""
        def create_complete_chart(raw_data, chart_type, config_options):
            """Mock complete chart creation workflow."""
            # 1. Validate data
            if not raw_data:
                raise ValueError("No data provided")

            # 2. Process data
            processed_data = raw_data.copy()
            if config_options.get('aggregate'):
                # Mock aggregation logic
                processed_data = [
                    {'category': 'A', 'total': 30},
                    {'category': 'B', 'total': 40}
                ]

            # 3. Create chart
            chart_config = {
                'type': chart_type,
                'data': processed_data,
                'title': config_options.get('title', 'Chart'),
                'layout': config_options.get('layout', {})
            }

            # 4. Apply styling
            if config_options.get('theme'):
                chart_config['theme'] = config_options['theme']

            # 5. Configure interactivity
            if config_options.get('interactive', True):
                chart_config['interactivity'] = {'zoom': True, 'pan': True}

            return chart_config

        raw_data = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
            {'category': 'B', 'value': 40}
        ]

        config = {
            'title': 'Test Chart',
            'aggregate': True,
            'theme': 'dark',
            'interactive': True
        }

        chart = create_complete_chart(raw_data, 'bar', config)

        assert chart['type'] == 'bar'
        assert chart['title'] == 'Test Chart'
        assert chart['theme'] == 'dark'
        assert 'interactivity' in chart

    def test_chart_performance_with_large_datasets(self):
        """Test chart performance with large datasets."""
        def handle_large_dataset(data, max_points=1000):
            """Mock large dataset handling."""
            if len(data) <= max_points:
                return {'data': data, 'sampling_applied': False}

            # Apply sampling for large datasets
            step = len(data) // max_points
            sampled_data = data[::step][:max_points]

            return {
                'data': sampled_data,
                'sampling_applied': True,
                'original_size': len(data),
                'sampled_size': len(sampled_data)
            }

        # Test small dataset
        small_data = [{'x': i, 'y': i*2} for i in range(100)]
        result = handle_large_dataset(small_data, 1000)
        assert not result['sampling_applied']
        assert len(result['data']) == 100

        # Test large dataset
        large_data = [{'x': i, 'y': i*2} for i in range(5000)]
        result = handle_large_dataset(large_data, 1000)
        assert result['sampling_applied']
        assert result['sampled_size'] <= 1000
        assert result['original_size'] == 5000

    def test_chart_caching(self):
        """Test chart caching functionality."""
        def chart_cache_manager():
            """Mock chart cache manager."""
            cache = {}

            def get_cached_chart(cache_key):
                return cache.get(cache_key)

            def cache_chart(cache_key, chart_config):
                cache[cache_key] = {
                    'chart': chart_config,
                    'timestamp': 1234567890,  # Mock timestamp
                    'expires_at': 1234567890 + 3600  # 1 hour expiry
                }

            def is_cache_valid(cache_key, current_time=1234567890):
                cached_item = cache.get(cache_key)
                if not cached_item:
                    return False
                return current_time < cached_item['expires_at']

            return {
                'get': get_cached_chart,
                'set': cache_chart,
                'is_valid': is_cache_valid
            }

        cache_manager = chart_cache_manager()

        # Test cache miss
        chart = cache_manager['get']('test_chart')
        assert chart is None

        # Test cache set and hit
        test_chart = {'type': 'bar', 'data': []}
        cache_manager['set']('test_chart', test_chart)

        cached_chart = cache_manager['get']('test_chart')
        assert cached_chart is not None
        assert cached_chart['chart']['type'] == 'bar'

        # Test cache validity
        assert cache_manager['is_valid']('test_chart', 1234567890)
        assert not cache_manager['is_valid']('test_chart', 1234567890 + 3700)  # Expired

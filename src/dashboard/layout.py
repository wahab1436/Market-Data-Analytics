"""
Dashboard Layout Module
Professional financial dashboard layout with Plotly Dash
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from typing import Dict, Any, List


def create_layout(config: Dict[str, Any], disclaimer_text: str) -> html.Div:
    """Create the main dashboard layout."""
    
    symbols = config['data']['symbols']
    color_palette = config['dashboard']['color_palette']
    
    # Current timestamp for display
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header with navigation
    header = html.Div([
        html.Div([
            html.Div([
                html.H1("Market Insight Platform", style={
                    'color': 'white',
                    'margin': '0',
                    'fontSize': '28px',
                    'fontWeight': '600',
                    'letterSpacing': '-0.5px'
                }),
                html.P("Advanced Analytics Dashboard", style={
                    'color': 'rgba(255,255,255,0.85)',
                    'margin': '5px 0 0 0',
                    'fontSize': '14px',
                    'fontWeight': '400'
                })
            ], style={'flex': '1'}),
            
            html.Div([
                html.Div(f"Last Update: {current_time}", style={
                    'color': 'rgba(255,255,255,0.75)',
                    'fontSize': '13px',
                    'padding': '8px 16px',
                    'background': 'rgba(255,255,255,0.1)',
                    'borderRadius': '6px',
                    'border': '1px solid rgba(255,255,255,0.15)'
                })
            ])
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '24px 32px',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-between'
        })
    ], style={
        'background': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'marginBottom': '0'
    })
    
    # Symbol selector and controls
    controls_card = html.Div([
        html.Div([
            html.H3("Controls", style={
                'fontSize': '18px',
                'fontWeight': '600',
                'marginBottom': '24px',
                'color': '#1e3c72',
                'borderBottom': '2px solid #e8f0fe',
                'paddingBottom': '12px'
            }),
            
            html.Div([
                html.Label("Select Asset", style={
                    'fontSize': '13px',
                    'fontWeight': '600',
                    'color': '#5f6368',
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.5px',
                    'marginBottom': '8px',
                    'display': 'block'
                }),
                dcc.Dropdown(
                    id="symbol-selector",
                    options=[{"label": s, "value": s} for s in symbols],
                    value=symbols[0] if symbols else None,
                    clearable=False,
                    style={
                        'marginBottom': '24px',
                        'fontSize': '14px'
                    }
                ),
            ]),
            
            html.Div([
                html.Label("Date Range", style={
                    'fontSize': '13px',
                    'fontWeight': '600',
                    'color': '#5f6368',
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.5px',
                    'marginBottom': '8px',
                    'display': 'block'
                }),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=config['data']['date_range']['start'],
                    end_date=config['data']['date_range']['end'],
                    display_format='YYYY-MM-DD',
                    style={'marginBottom': '24px'}
                ),
            ]),
            
            html.Div([
                html.Label("Analysis Focus", style={
                    'fontSize': '13px',
                    'fontWeight': '600',
                    'color': '#5f6368',
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.5px',
                    'marginBottom': '12px',
                    'display': 'block'
                }),
                dcc.RadioItems(
                    id="analysis-focus",
                    options=[
                        {'label': 'Price & Trends', 'value': 'price'},
                        {'label': 'Risk & Volatility', 'value': 'volatility'},
                        {'label': 'Volume & Liquidity', 'value': 'volume'},
                        {'label': 'Pattern Analysis', 'value': 'similarity'}
                    ],
                    value='price',
                    labelStyle={
                        'display': 'block',
                        'marginBottom': '10px',
                        'fontSize': '14px',
                        'color': '#202124'
                    },
                    style={'marginBottom': '24px'}
                ),
            ]),
            
            html.Button(
                "Refresh Data",
                id="refresh-button",
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '12px 24px',
                    'fontSize': '14px',
                    'fontWeight': '600',
                    'color': 'white',
                    'background': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
                    'border': 'none',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }
            ),
        ], style={
            'background': 'white',
            'borderRadius': '12px',
            'padding': '24px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
            'border': '1px solid #e8eaed'
        })
    ])
    
    # KPI Cards
    kpi_cards = html.Div([
        create_kpi_card("Current Price", "$--", "#1e3c72", "price-kpi"),
        create_kpi_card("Daily Return", "--%", "#0f9d58", "return-kpi"),
        create_kpi_card("Volatility", "--%", "#f4b400", "volatility-kpi"),
        create_kpi_card("Volume", "--", "#4285f4", "volume-kpi"),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(4, 1fr)',
        'gap': '20px',
        'marginBottom': '32px'
    })
    
    # Main content sections
    content = html.Div([
        # Price Analysis Section
        html.Div([
            html.Div([
                html.H2("Price Analysis", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Comprehensive price trends and technical indicators", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-price-chart",
                    type="circle",
                    color="#1e3c72",
                    children=dcc.Graph(
                        id="price-chart",
                        config={'displayModeBar': False},
                        style={'height': '500px'}
                    )
                )
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            }),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Key Insights", style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': '#202124',
                            'marginBottom': '16px'
                        }),
                        html.Div(id="price-insights", style={'fontSize': '14px', 'color': '#5f6368'})
                    ], style={
                        'background': 'white',
                        'borderRadius': '12px',
                        'padding': '24px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                        'border': '1px solid #e8eaed',
                        'height': '100%'
                    })
                ], style={'flex': '1', 'marginRight': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H3("Performance Metrics", style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': '#202124',
                            'marginBottom': '16px'
                        }),
                        html.Div(id="trend-metrics", style={'fontSize': '14px'})
                    ], style={
                        'background': 'white',
                        'borderRadius': '12px',
                        'padding': '24px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                        'border': '1px solid #e8eaed',
                        'height': '100%'
                    })
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '0'})
        ], id="price-section", style={'marginBottom': '48px'}),
        
        # Volatility Analysis Section
        html.Div([
            html.Div([
                html.H2("Volatility Analysis", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Risk assessment and return distribution", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Div([
                    dcc.Loading(
                        id="loading-volatility-chart",
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="volatility-chart",
                            config={'displayModeBar': False},
                            style={'height': '450px'}
                        )
                    )
                ], style={'flex': '2', 'marginRight': '20px'}),
                
                html.Div([
                    dcc.Loading(
                        id="loading-volatility-distribution",
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="volatility-distribution-chart",
                            config={'displayModeBar': False},
                            style={'height': '450px'}
                        )
                    )
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed',
                'gap': '0'
            }),
            
            html.Div([
                html.H3("Volatility Insights", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '16px'
                }),
                html.Div(id="volatility-insights", style={'fontSize': '14px', 'color': '#5f6368'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            })
        ], id="volatility-section", style={'marginBottom': '48px'}),
        
        # Volume Analysis Section
        html.Div([
            html.Div([
                html.H2("Volume Analysis", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Trading volume and liquidity metrics", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-volume-chart",
                    type="circle",
                    color="#1e3c72",
                    children=dcc.Graph(
                        id="volume-chart",
                        config={'displayModeBar': False},
                        style={'height': '500px'}
                    )
                )
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            }),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Volume-Price Relationship", style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': '#202124',
                            'marginBottom': '16px'
                        }),
                        html.Div(id="volume-relationship", style={'fontSize': '14px'})
                    ], style={
                        'background': 'white',
                        'borderRadius': '12px',
                        'padding': '24px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                        'border': '1px solid #e8eaed',
                        'height': '100%'
                    })
                ], style={'flex': '1', 'marginRight': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H3("Volume Insights", style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': '#202124',
                            'marginBottom': '16px'
                        }),
                        html.Div(id="volume-insights", style={'fontSize': '14px', 'color': '#5f6368'})
                    ], style={
                        'background': 'white',
                        'borderRadius': '12px',
                        'padding': '24px',
                        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                        'border': '1px solid #e8eaed',
                        'height': '100%'
                    })
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '0'})
        ], id="volume-section", style={'marginBottom': '48px'}),
        
        # Similarity Analysis Section
        html.Div([
            html.Div([
                html.H2("Pattern Similarity", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Historical pattern matching and analogous periods", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-similarity-chart",
                    type="circle",
                    color="#1e3c72",
                    children=dcc.Graph(
                        id="similarity-chart",
                        config={'displayModeBar': False},
                        style={'height': '450px'}
                    )
                )
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            }),
            
            html.Div([
                html.H3("Historical Analogs", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '16px'
                }),
                html.Div(id="analog-insights", style={'fontSize': '14px', 'color': '#5f6368'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            })
        ], id="similarity-section", style={'marginBottom': '48px'}),
        
        # Machine Learning Models Section
        html.Div([
            html.Div([
                html.H2("Predictive Models", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Machine learning forecasts and performance analysis", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                dcc.Tabs(
                    id="model-tabs",
                    value="regression-tab",
                    children=[
                        dcc.Tab(
                            label="Regression Models",
                            value="regression-tab",
                            style=tab_style,
                            selected_style=tab_selected_style
                        ),
                        dcc.Tab(
                            label="KNN Similarity",
                            value="knn-tab",
                            style=tab_style,
                            selected_style=tab_selected_style
                        ),
                        dcc.Tab(
                            label="XGBoost Predictions",
                            value="xgboost-tab",
                            style=tab_style,
                            selected_style=tab_selected_style
                        ),
                    ],
                    style={
                        'marginBottom': '0',
                        'borderBottom': '2px solid #e8eaed'
                    }
                ),
                
                html.Div(id="model-tab-content", style={
                    'padding': '32px 24px',
                    'background': 'white'
                })
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed',
                'overflow': 'hidden'
            })
        ], id="models-section", style={'marginBottom': '48px'}),
        
        # Explainability Section
        html.Div([
            html.Div([
                html.H2("Model Explainability", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Feature importance and model interpretability", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-explainability-chart",
                    type="circle",
                    color="#1e3c72",
                    children=dcc.Graph(
                        id="explainability-chart",
                        config={'displayModeBar': False},
                        style={'height': '450px'}
                    )
                )
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            }),
            
            html.Div([
                html.H3("Model Insights", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '16px'
                }),
                html.Div(id="model-insights", style={'fontSize': '14px', 'color': '#5f6368'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            })
        ], id="explainability-section", style={'marginBottom': '48px'}),
        
        # Cross-Symbol Analysis Section
        html.Div([
            html.Div([
                html.H2("Cross-Symbol Analysis", style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '4px'
                }),
                html.P("Comparative performance and correlation analysis", style={
                    'fontSize': '14px',
                    'color': '#5f6368',
                    'margin': '0'
                })
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Div([
                    dcc.Loading(
                        id="loading-correlation-chart",
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="correlation-chart",
                            config={'displayModeBar': False},
                            style={'height': '450px'}
                        )
                    )
                ], style={'flex': '1', 'marginRight': '20px'}),
                
                html.Div([
                    dcc.Loading(
                        id="loading-performance-chart",
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="performance-chart",
                            config={'displayModeBar': False},
                            style={'height': '450px'}
                        )
                    )
                ], style={'flex': '1'})
            ], style={
                'display': 'flex',
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed',
                'gap': '0'
            }),
            
            html.Div([
                html.H3("Cross-Symbol Insights", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '16px'
                }),
                html.Div(id="cross-symbol-insights", style={'fontSize': '14px', 'color': '#5f6368'})
            ], style={
                'background': 'white',
                'borderRadius': '12px',
                'padding': '24px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e8eaed'
            })
        ], id="cross-symbol-section", style={'marginBottom': '48px'}),
    ])
    
    # Footer
    footer = html.Div([
        html.Div([
            html.Div([
                html.H4("Disclaimer", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '12px'
                }),
                html.P(disclaimer_text, style={
                    'fontSize': '13px',
                    'color': '#5f6368',
                    'lineHeight': '1.6',
                    'marginBottom': '8px'
                }),
                html.P([
                    "This platform is for demonstration and educational purposes only. ",
                    html.Strong("Not financial advice. "),
                    "All data may be delayed or inaccurate. Past performance does not guarantee future results."
                ], style={
                    'fontSize': '13px',
                    'color': '#5f6368',
                    'lineHeight': '1.6'
                })
            ], style={'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.Span("Market Insight Platform v1.0", style={'marginRight': '20px'}),
                    html.Span("Python 3.13", style={'marginRight': '20px'}),
                    html.Span("Plotly Dash", style={'marginRight': '20px'}),
                    html.Span(f"Updated: {current_time}")
                ], style={
                    'fontSize': '12px',
                    'color': '#80868b',
                    'paddingTop': '16px',
                    'borderTop': '1px solid #e8eaed'
                })
            ])
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '32px'
        })
    ], style={
        'background': '#f8f9fa',
        'marginTop': '48px',
        'borderTop': '1px solid #e8eaed'
    })
    
    # Main layout
    layout = html.Div([
        # Store preloaded data
        dcc.Store(id='analysis-data-store'),
        dcc.Store(id='model-data-store'),
        dcc.Store(id='featured-data-store'),
        dcc.Location(id='url', refresh=False),
        
        # Header
        header,
        
        # Main container
        html.Div([
            html.Div([
                # Sidebar controls
                html.Div(controls_card, style={'marginRight': '32px', 'width': '300px', 'flexShrink': '0'}),
                
                # Main content
                html.Div([
                    # KPI Cards
                    kpi_cards,
                    
                    # Main content
                    content
                ], style={'flex': '1', 'minWidth': '0'})
            ], style={
                'display': 'flex',
                'maxWidth': '1400px',
                'margin': '32px auto',
                'padding': '0 32px'
            }),
            
            # Footer
            footer
            
        ], style={'background': '#f8f9fa', 'minHeight': '100vh'})
    ], style={
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
        'margin': '0',
        'padding': '0'
    })
    
    return layout


def create_kpi_card(title: str, value: str, color: str, kpi_id: str) -> html.Div:
    """Create a KPI card component."""
    return html.Div([
        html.Div([
            html.Div(title, style={
                'fontSize': '13px',
                'fontWeight': '600',
                'color': '#5f6368',
                'textTransform': 'uppercase',
                'letterSpacing': '0.5px',
                'marginBottom': '12px'
            }),
            html.Div(id=kpi_id, children=value, style={
                'fontSize': '32px',
                'fontWeight': '700',
                'color': '#202124',
                'marginBottom': '8px',
                'lineHeight': '1'
            }),
            html.Div(id=f"{kpi_id}-trend", style={
                'fontSize': '13px',
                'fontWeight': '500',
                'color': '#5f6368'
            })
        ])
    ], style={
        'background': 'white',
        'borderRadius': '12px',
        'padding': '24px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
        'border': '1px solid #e8eaed',
        'borderLeft': f'4px solid {color}',
        'transition': 'transform 0.2s ease, box-shadow 0.2s ease'
    })


def create_model_tab_content(tab_id: str) -> html.Div:
    """Create content for model tabs."""
    if tab_id == "regression-tab":
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Loading(
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="regression-chart",
                            config={'displayModeBar': False},
                            style={'height': '400px'}
                        )
                    )
                ], style={'flex': '2', 'marginRight': '20px'}),
                
                html.Div([
                    dcc.Loading(
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="regression-metrics-chart",
                            config={'displayModeBar': False},
                            style={'height': '400px'}
                        )
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'marginBottom': '24px', 'gap': '0'}),
            
            html.Div([
                html.H3("Regression Model Insights", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '16px'
                }),
                html.Div(id="regression-insights", style={'fontSize': '14px', 'color': '#5f6368'})
            ], style={
                'background': '#f8f9fa',
                'borderRadius': '8px',
                'padding': '20px',
                'border': '1px solid #e8eaed'
            })
        ])
    
    elif tab_id == "knn-tab":
        return html.Div([
            html.Div([
                dcc.Loading(
                    type="circle",
                    color="#1e3c72",
                    children=dcc.Graph(
                        id="knn-chart",
                        config={'displayModeBar': False},
                        style={'height': '400px'}
                    )
                )
            ], style={'marginBottom': '24px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Historical Analogs", style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': '#202124',
                            'marginBottom': '16px'
                        }),
                        html.Div(id="knn-analogs", style={'fontSize': '14px'})
                    ], style={
                        'background': '#f8f9fa',
                        'borderRadius': '8px',
                        'padding': '20px',
                        'border': '1px solid #e8eaed',
                        'height': '100%'
                    })
                ], style={'flex': '1', 'marginRight': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H3("KNN Insights", style={
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'color': '#202124',
                            'marginBottom': '16px'
                        }),
                        html.Div(id="knn-insights", style={'fontSize': '14px', 'color': '#5f6368'})
                    ], style={
                        'background': '#f8f9fa',
                        'borderRadius': '8px',
                        'padding': '20px',
                        'border': '1px solid #e8eaed',
                        'height': '100%'
                    })
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '0'})
        ])
    
    elif tab_id == "xgboost-tab":
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Loading(
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="xgboost-prediction-chart",
                            config={'displayModeBar': False},
                            style={'height': '400px'}
                        )
                    )
                ], style={'flex': '2', 'marginRight': '20px'}),
                
                html.Div([
                    dcc.Loading(
                        type="circle",
                        color="#1e3c72",
                        children=dcc.Graph(
                            id="xgboost-importance-chart",
                            config={'displayModeBar': False},
                            style={'height': '400px'}
                        )
                    )
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'marginBottom': '24px', 'gap': '0'}),
            
            html.Div([
                html.H3("XGBoost Performance Metrics", style={
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'color': '#202124',
                    'marginBottom': '16px'
                }),
                html.Div(id="xgboost-metrics", style={'fontSize': '14px'})
            ], style={
                'background': '#f8f9fa',
                'borderRadius': '8px',
                'padding': '20px',
                'border': '1px solid #e8eaed'
            })
        ])
    
    return html.Div("Select a model tab to view analysis.", style={
        'fontSize': '14px',
        'color': '#5f6368',
        'padding': '40px',
        'textAlign': 'center'
    })


# Tab styles
tab_style = {
    'padding': '16px 24px',
    'fontSize': '14px',
    'fontWeight': '500',
    'border': 'none',
    'borderBottom': '3px solid transparent',
    'backgroundColor': 'white',
    'color': '#5f6368',
    'cursor': 'pointer',
    'transition': 'all 0.3s ease'
}

tab_selected_style = {
    'padding': '16px 24px',
    'fontSize': '14px',
    'fontWeight': '600',
    'border': 'none',
    'borderBottom': '3px solid #1e3c72',
    'backgroundColor': 'white',
    'color': '#1e3c72'
}
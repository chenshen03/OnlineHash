function plot_recall_prec(recall, precision, opts)

if nargin < 3
    opts.methodID = 'OnlineHash';
    opts.dataset = 'cifar';
    opts.nbits = 64;
    opts.dirs.exp = '../cachedir';
end

% plot attribution
hashmethods = opts.methodID;
db_name = opts.dataset;
line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;

%% show precision vs. recall , i is the selection of which bits.
figure('Color', [1 1 1]); hold on;
figure('visible','off');

p = plot(recall, precision);
set(p,'Color', 'b')
set(p,'Marker', '*');
set(p,'LineWidth', line_width);
set(p,'MarkerSize', marker_size);

str_nbits = num2str(opts.nbits);
h1 = xlabel('Recall');
h2 = ylabel('Precision');
title([db_name, ' @ ', str_nbits, ' bits'], 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
axis square;
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
set(hleg,'Location', 'best');
set(gca, 'linewidth', linewidth);
box on;
grid on;
hold off;

saveas(p, sprintf('%s/precision_recall.jpg', opts.dirs.exp));
